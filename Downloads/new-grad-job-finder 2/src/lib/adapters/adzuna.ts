import { JobSourceAdapter } from "@/lib/adapters/types";
import { LOCATION_SEARCH_TERMS } from "@/lib/location/locationAliases";
import { RawJobListing, SearchParams, TargetMarket } from "@/lib/types";

// Adzuna aggregates postings from thousands of companies' own career sites
// and other job boards - this is the "broad recall" tier that replaces
// needing to know any individual company's ATS. Free tier requires a
// registered app_id/app_key (https://developer.adzuna.com/), set via env.
const ADZUNA_APP_ID = process.env.ADZUNA_APP_ID;
const ADZUNA_APP_KEY = process.env.ADZUNA_APP_KEY;

// Adzuna's "where" param wants a plain city/region string.
const ADZUNA_LOCATION: Record<TargetMarket, string> = {
  "New York City": "New York",
  "San Francisco Bay Area": "San Francisco",
  Chicago: "Chicago",
  Seattle: "Seattle",
  "North Carolina": "Raleigh",
};

type AdzunaJob = {
  id: string;
  title: string;
  company?: { display_name?: string };
  location?: { display_name?: string };
  redirect_url: string;
  created?: string;
  description?: string;
};

export class AdzunaAdapter implements JobSourceAdapter {
  sourceName = "Adzuna";

  async isAvailable(): Promise<boolean> {
    return Boolean(ADZUNA_APP_ID && ADZUNA_APP_KEY);
  }

  async searchJobs(params: SearchParams): Promise<RawJobListing[]> {
    if (!ADZUNA_APP_ID || !ADZUNA_APP_KEY) return [];

    const locations = params.locations.length > 0 ? params.locations : (Object.keys(ADZUNA_LOCATION) as TargetMarket[]);
    const earlyCareerTerms = "new grad,entry level,university graduate,early career";
    const roleTerms = params.keywords.join(" ");

    const results: RawJobListing[] = [];

    await Promise.all(
      locations.map(async (market) => {
        const where = ADZUNA_LOCATION[market];
        const url = new URL(`https://api.adzuna.com/v1/api/jobs/us/search/1`);
        url.searchParams.set("app_id", ADZUNA_APP_ID);
        url.searchParams.set("app_key", ADZUNA_APP_KEY);
        url.searchParams.set("results_per_page", "30");
        url.searchParams.set("where", where);
        url.searchParams.set("what_or", earlyCareerTerms);
        if (roleTerms) url.searchParams.set("what_and", roleTerms);
        url.searchParams.set("max_days_old", "45");

        try {
          const res = await fetch(url.toString(), { cache: "no-store" });
          if (!res.ok) return;
          const data = (await res.json()) as { results: AdzunaJob[] };

          for (const job of data.results ?? []) {
            results.push({
              externalId: job.id,
              source: this.sourceName,
              sourceUrl: job.redirect_url,
              applicationUrl: job.redirect_url,
              companyName: job.company?.display_name ?? "Unknown company",
              title: job.title,
              description: job.description,
              rawLocation: job.location?.display_name,
              postedAt: job.created,
            });
          }
        } catch {
          // A single location failing shouldn't fail the whole adapter.
        }
      }),
    );

    return results;
  }
}
