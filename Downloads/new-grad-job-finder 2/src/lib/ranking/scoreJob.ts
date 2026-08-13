import {
  classifyRoleCategories,
  classifySeniority,
  extractMaxExperienceYears,
  hasSeniorityExclusionTerms,
  isFintechRelevant,
  isRiskRelevant,
} from "@/lib/ranking/classifyJob";
import { RoleCategory, SearchParams, SeniorityLevel, TargetMarket } from "@/lib/types";

export type ScoreInput = {
  title: string;
  description?: string;
  region?: TargetMarket;
  postedAt?: string;
  isPriorityCompany: boolean;
  isStaleOrUnverified: boolean;
  employmentType?: string;
};

export type ScoreResult = {
  score: number;
  reasons: string[];
  roleCategories: RoleCategory[];
  seniorityLevel: SeniorityLevel;
  isFintechRelevant: boolean;
  isRiskRelevant: boolean;
  isExcluded: boolean;
  exclusionReason?: string;
};

/**
 * Implements the ranking formula from spec section 7. Seniority-exclusion
 * terms hard-exclude the role (rather than just applying -100) - this was
 * an explicit simplification decision for readability: an excluded role
 * disappears instead of confusingly sitting at the bottom of the list.
 */
export function scoreJob(input: ScoreInput, params: SearchParams): ScoreResult {
  const roleCategories = classifyRoleCategories(input.title, input.description);
  const seniorityLevel = classifySeniority(input.title, input.description);
  const fintechRelevant = isFintechRelevant(input.title, input.description);
  const riskRelevant = isRiskRelevant(input.title, input.description);

  // --- Hard exclusions ---
  if (hasSeniorityExclusionTerms(input.title)) {
    return {
      score: 0,
      reasons: [],
      roleCategories,
      seniorityLevel,
      isFintechRelevant: fintechRelevant,
      isRiskRelevant: riskRelevant,
      isExcluded: true,
      exclusionReason: "Title contains a senior/staff/lead/manager-level term.",
    };
  }

  const requiredYears = extractMaxExperienceYears(input.description);
  if (requiredYears !== undefined && requiredYears > params.maxExperienceYears + 2) {
    // Only hard-exclude when it's clearly beyond new-grad range (>4 yrs
    // by default); 3-4 years just takes the -25 penalty below instead.
    return {
      score: 0,
      reasons: [],
      roleCategories,
      seniorityLevel,
      isFintechRelevant: fintechRelevant,
      isRiskRelevant: riskRelevant,
      isExcluded: true,
      exclusionReason: `Requires ${requiredYears}+ years of experience.`,
    };
  }

  if (input.employmentType?.toLowerCase().includes("contract") && !params.includeContractRoles) {
    return {
      score: 0,
      reasons: [],
      roleCategories,
      seniorityLevel,
      isFintechRelevant: fintechRelevant,
      isRiskRelevant: riskRelevant,
      isExcluded: true,
      exclusionReason: "Contract-only role (enable \"Include contract roles\" to see these).",
    };
  }

  // --- Additive scoring ---
  let score = 0;
  const reasons: string[] = [];

  if (seniorityLevel === "new_grad") {
    score += 30;
    reasons.push("new-grad program language");
  } else if (seniorityLevel === "early_career") {
    score += 20;
    reasons.push("early-career language");
  }

  if (seniorityLevel === "entry_level" || (requiredYears !== undefined && requiredYears <= 2)) {
    score += 20;
    reasons.push("entry-level title or 0-2 years experience");
  }

  const matchesCategory = roleCategories.some((c) => params.categories.includes(c));
  if (matchesCategory) {
    score += 15;
    reasons.push(`matches selected category (${roleCategories[0]})`);
  }

  if (input.region && params.locations.includes(input.region)) {
    score += 15;
    reasons.push(`${input.region} location`);
  }

  if (input.isPriorityCompany) {
    score += 10;
    reasons.push("priority company");
  }

  if (fintechRelevant) {
    score += 5;
    reasons.push("fintech relevance");
  }

  if (riskRelevant) {
    score += 5;
    reasons.push("risk relevance");
  }

  if (input.postedAt) {
    const daysSincePosted = (Date.now() - new Date(input.postedAt).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSincePosted <= 7) {
      score += 5;
      reasons.push(`posted ${Math.max(0, Math.round(daysSincePosted))} days ago`);
    }
  }

  if (requiredYears !== undefined && requiredYears > params.maxExperienceYears) {
    score -= 25;
    reasons.push(`requires more than ${params.maxExperienceYears} years experience`);
  }

  if (input.isStaleOrUnverified) {
    score -= 50;
    reasons.push("listing could not be verified as active");
  }

  score = Math.max(0, Math.min(100, score));

  return {
    score,
    reasons,
    roleCategories,
    seniorityLevel,
    isFintechRelevant: fintechRelevant,
    isRiskRelevant: riskRelevant,
    isExcluded: false,
  };
}
