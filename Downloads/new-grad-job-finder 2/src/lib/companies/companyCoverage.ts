import { NormalizedJob } from "@/lib/types";

export type CompanyCoverage = {
  covered: string[]; // watched companies that appeared in results
  gaps: string[]; // watched companies with zero matching results
};

function normalize(name: string): string {
  return name.trim().toLowerCase().replace(/[.,]/g, "");
}

/**
 * Diffs the user's watched companies (their priority list) against the
 * companies that actually showed up in aggregator results. Anything with
 * zero matches is a "gap" - too small or enterprise-specific for the
 * broad aggregator APIs to reliably surface, and the right candidate for
 * a quick manual check (e.g. via Claude in Chrome) instead of building
 * bespoke per-company scraping.
 */
export function computeCompanyCoverage(
  jobs: NormalizedJob[],
  watchedCompanies: string[],
): CompanyCoverage {
  const resultCompanyNames = new Set(jobs.map((j) => normalize(j.companyName)));

  const covered: string[] = [];
  const gaps: string[] = [];

  for (const company of watchedCompanies) {
    const normalized = normalize(company);
    // Substring match in both directions - aggregator company names are
    // sometimes formatted differently ("JPMorgan Chase & Co." vs
    // "JPMorgan Chase"), so an exact match would under-count coverage.
    const isCovered = Array.from(resultCompanyNames).some(
      (resultName) => resultName.includes(normalized) || normalized.includes(resultName),
    );
    if (isCovered) {
      covered.push(company);
    } else {
      gaps.push(company);
    }
  }

  return { covered, gaps };
}
