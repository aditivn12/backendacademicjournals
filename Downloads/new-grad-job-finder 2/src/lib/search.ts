import { AdzunaAdapter } from "@/lib/adapters/adzuna";
import { MockAdapter } from "@/lib/adapters/mock";
import { TheMuseAdapter } from "@/lib/adapters/theMuse";
import { JobSourceAdapter } from "@/lib/adapters/types";
import { buildSmartLinks, SmartLinks } from "@/lib/companies/smartLinks";
import { computeCompanyCoverage } from "@/lib/companies/companyCoverage";
import { deduplicateJobs } from "@/lib/ranking/deduplicateJobs";
import { normalizeJob } from "@/lib/ranking/normalizeJob";
import { NormalizedJob, RawJobListing, SearchParams, SourceQueryResult } from "@/lib/types";

export type SearchResponse = {
  searchedAt: string;
  resultCount: number;
  jobs: NormalizedJob[];
  sourcesQueried: SourceQueryResult[];
  warnings: string[];
  gapCompanies: SmartLinks[];
};

const CACHE_MINUTES = parseInt(process.env.SEARCH_CACHE_MINUTES ?? "30", 10);
const USE_MOCK = process.env.USE_MOCK_DATA === "true";

// Simple in-memory cache - fine for a single-user, single-process app.
// Resets on server restart, which is acceptable for personal use.
const cache = new Map<string, { cachedAt: number; response: SearchResponse }>();

function getAdapters(): JobSourceAdapter[] {
  if (USE_MOCK) return [new MockAdapter()];
  return [new AdzunaAdapter(), new TheMuseAdapter()];
}

function cacheKey(params: SearchParams): string {
  const { forceRefresh, ...cacheable } = params;
  return JSON.stringify(cacheable);
}

export async function runSearch(params: SearchParams): Promise<SearchResponse> {
  const key = cacheKey(params);

  if (!params.forceRefresh) {
    const hit = cache.get(key);
    if (hit && Date.now() - hit.cachedAt < CACHE_MINUTES * 60 * 1000) {
      return hit.response;
    }
  }

  const adapters = getAdapters();
  const sourcesQueried: SourceQueryResult[] = [];
  const warnings: string[] = [];
  const rawJobs: RawJobListing[] = [];

  await Promise.all(
    adapters.map(async (adapter) => {
      try {
        const available = await adapter.isAvailable();
        if (!available) {
          sourcesQueried.push({
            source: adapter.sourceName,
            status: "unavailable",
            jobsFound: 0,
            message: "No companies configured for this source yet.",
          });
          return;
        }

        const jobs = await adapter.searchJobs(params);
        rawJobs.push(...jobs);
        sourcesQueried.push({
          source: adapter.sourceName,
          status: "success",
          jobsFound: jobs.length,
          message: `${adapter.sourceName} search completed.`,
        });
      } catch (err) {
        sourcesQueried.push({
          source: adapter.sourceName,
          status: "error",
          jobsFound: 0,
          message: `${adapter.sourceName} was temporarily unavailable.`,
        });
        warnings.push(`One source (${adapter.sourceName}) was temporarily unavailable.`);
      }
    }),
  );

  const prioritySet = new Set(params.priorityCompanies.map((c) => c.trim().toLowerCase()));
  const normalized = rawJobs.map((job) => normalizeJob(job, params, prioritySet));

  const filtered = normalized.filter((job) => {
    if (job.isExcluded) return false;
    if (!params.includeRemote && job.workArrangement === "remote") return false;
    if (!params.includeHybrid && job.workArrangement === "hybrid") return false;
    if (
      params.excludedKeywords.length > 0 &&
      params.excludedKeywords.some((kw) =>
        `${job.title} ${job.description ?? ""}`.toLowerCase().includes(kw.toLowerCase()),
      )
    ) {
      return false;
    }
    if (params.locations.length > 0 && job.region && !params.locations.includes(job.region)) {
      return false;
    }
    if (
      params.categories.length > 0 &&
      !job.roleCategories.some((c) => params.categories.includes(c))
    ) {
      return false;
    }
    return true;
  });

  const deduped = deduplicateJobs(filtered);
  deduped.sort((a, b) => b.matchScore - a.matchScore);

  if (rawJobs.length === 0 && !USE_MOCK) {
    warnings.push(
      "No results came back from the aggregator sources. Check that ADZUNA_APP_ID/ADZUNA_APP_KEY are set in .env, or set USE_MOCK_DATA=true to develop against seeded sample data.",
    );
  }

  const { gaps } = computeCompanyCoverage(deduped, params.priorityCompanies);
  const gapCompanies = gaps.map((company) =>
    buildSmartLinks(company, params.categories, params.locations),
  );

  const response: SearchResponse = {
    searchedAt: new Date().toISOString(),
    resultCount: deduped.length,
    jobs: deduped,
    sourcesQueried,
    warnings,
    gapCompanies,
  };

  cache.set(key, { cachedAt: Date.now(), response });
  return response;
}
