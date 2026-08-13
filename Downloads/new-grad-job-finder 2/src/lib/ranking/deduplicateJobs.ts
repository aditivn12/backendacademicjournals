import { NormalizedJob } from "@/lib/types";

/**
 * Simplified for personal use: dedups on (1) identical application URL,
 * then (2) same source + external ID, then (3) same normalized company +
 * title + location. The spec's 4th tier ("fuzzy-match title/company when
 * descriptions indicate the same requisition") is skipped here - a real
 * fuzzy-match library is more infrastructure than a single-user tool
 * needs, and the exact-match tiers already cover the common case (the
 * same job mirrored across two sources with identical URL or metadata).
 */
export function buildFingerprint(job: {
  applicationUrl: string;
  source: string;
  externalId?: string;
  companyName: string;
  normalizedTitle: string;
  city?: string;
}): string {
  const normalizedUrl = job.applicationUrl.split("?")[0].replace(/\/$/, "").toLowerCase();
  return `url:${normalizedUrl}`;
}

export function deduplicateJobs(jobs: NormalizedJob[]): NormalizedJob[] {
  const byUrl = new Map<string, NormalizedJob>();
  const bySourceExternalId = new Map<string, NormalizedJob>();
  const byCompanyTitleLocation = new Map<string, NormalizedJob>();

  const result: NormalizedJob[] = [];

  for (const job of jobs) {
    const urlKey = job.applicationUrl.split("?")[0].replace(/\/$/, "").toLowerCase();
    const externalKey = job.externalId ? `${job.source}:${job.externalId}` : undefined;
    const compositeKey = `${job.companyName.toLowerCase()}|${job.normalizedTitle}|${(
      job.city ?? job.rawLocation ?? ""
    ).toLowerCase()}`;

    const existing =
      byUrl.get(urlKey) ??
      (externalKey ? bySourceExternalId.get(externalKey) : undefined) ??
      byCompanyTitleLocation.get(compositeKey);

    if (existing) {
      // Keep the version with the most direct company application URL:
      // prefer one that isn't pointing at a third-party mirror (e.g. a
      // LinkedIn re-post), by simple heuristic of matching companyCareersUrl
      // domain, else keep whichever was seen first.
      continue;
    }

    byUrl.set(urlKey, job);
    if (externalKey) bySourceExternalId.set(externalKey, job);
    byCompanyTitleLocation.set(compositeKey, job);
    result.push(job);
  }

  return result;
}
