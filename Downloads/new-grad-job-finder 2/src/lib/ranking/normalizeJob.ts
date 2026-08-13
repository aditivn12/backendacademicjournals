import { inferWorkArrangement, normalizeLocation } from "@/lib/location/normalizeLocation";
import { scoreJob } from "@/lib/ranking/scoreJob";
import { NormalizedJob, RawJobListing, SearchParams } from "@/lib/types";

function normalizeTitle(title: string): string {
  return title.trim().toLowerCase().replace(/\s+/g, " ");
}

export function normalizeJob(
  raw: RawJobListing,
  params: SearchParams,
  priorityCompanySet: Set<string>,
): NormalizedJob {
  const location = normalizeLocation(raw.rawLocation);
  const workArrangement = inferWorkArrangement(raw.rawLocation, raw.description);
  const normalizedTitle = normalizeTitle(raw.title);
  const isPriorityCompany = priorityCompanySet.has(raw.companyName.toLowerCase());

  // A listing is only "unverified" if it's missing a usable application
  // URL - we don't make live HEAD requests to every result by default
  // since that would multiply outbound requests; source adapters that can
  // cheaply confirm a URL (e.g. a 200 from the ATS API itself) are treated
  // as pre-verified.
  const isStaleOrUnverified = !raw.applicationUrl;

  const scored = scoreJob(
    {
      title: raw.title,
      description: raw.description,
      region: location.region,
      postedAt: raw.postedAt,
      isPriorityCompany,
      isStaleOrUnverified,
      employmentType: raw.employmentType,
    },
    params,
  );

  return {
    externalId: raw.externalId,
    source: raw.source,
    sourceUrl: raw.sourceUrl,
    applicationUrl: raw.applicationUrl,
    urlVerifiedAt: raw.applicationUrl ? new Date().toISOString() : undefined,

    companyName: raw.companyName,
    companyCareersUrl: raw.companyCareersUrl,

    title: raw.title,
    normalizedTitle,
    description: raw.description,
    employmentType: raw.employmentType,
    workArrangement,

    rawLocation: raw.rawLocation,
    city: location.city,
    state: location.state,
    region: location.region,
    country: location.region ? "US" : undefined,

    roleCategories: scored.roleCategories,
    seniorityLevel: scored.seniorityLevel,

    postedAt: raw.postedAt,
    retrievedAt: new Date().toISOString(),

    matchScore: scored.score,
    matchReasons: scored.reasons,

    isPriorityCompany,
    isFintechRelevant: scored.isFintechRelevant,
    isRiskRelevant: scored.isRiskRelevant,

    fingerprint: `${raw.source}:${raw.externalId ?? raw.applicationUrl}`,
    isExcluded: scored.isExcluded,
    exclusionReason: scored.exclusionReason,
  };
}
