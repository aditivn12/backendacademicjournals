import { SEEDED_COMPANIES } from "@/lib/companies/seededCompanies";
import { CompanySuggestion, RoleCategory, TargetMarket } from "@/lib/types";

const FINTECH_RISK_CATEGORIES: RoleCategory[] = [
  "Fintech / Financial Services",
  "Risk Management / Risk Technology",
  "Quantitative / Trading Technology",
];

export function recommendCompanies(
  locations: TargetMarket[],
  categories: RoleCategory[],
  priorityCompanies: string[],
): CompanySuggestion[] {
  const prioritySet = new Set(priorityCompanies.map((c) => c.trim().toLowerCase()));
  const wantsFintechOrRisk = categories.some((c) => FINTECH_RISK_CATEGORIES.includes(c));

  const candidates = SEEDED_COMPANIES.filter((company) => {
    // Already in the user's own priority list - don't re-suggest it.
    if (prioritySet.has(company.name.toLowerCase())) return false;

    const locationMatch =
      locations.length === 0 || company.regions.some((r) => locations.includes(r));
    if (!locationMatch) return false;

    return true;
  });

  // Rank: fintech/risk relevance (if the user cares about it) + new-grad
  // pipeline first, then everything else, alphabetically within tiers.
  const ranked = candidates.sort((a, b) => {
    const aScore =
      (wantsFintechOrRisk && (a.isFintechRelevant || a.isRiskRelevant) ? 2 : 0) +
      (a.hasNewGradPipeline ? 1 : 0);
    const bScore =
      (wantsFintechOrRisk && (b.isFintechRelevant || b.isRiskRelevant) ? 2 : 0) +
      (b.hasNewGradPipeline ? 1 : 0);
    if (bScore !== aScore) return bScore - aScore;
    return a.name.localeCompare(b.name);
  });

  return ranked.slice(0, 12).map((company) => ({
    name: company.name,
    industry: company.industry,
    regions: company.regions,
    reason: company.recommendationNote,
    hasNewGradPipeline: company.hasNewGradPipeline,
    careersUrl: company.careersUrl,
    isUserPriority: false,
  }));
}
