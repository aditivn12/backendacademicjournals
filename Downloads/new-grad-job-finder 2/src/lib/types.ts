export const TARGET_MARKETS = [
  "New York City",
  "San Francisco Bay Area",
  "Chicago",
  "Seattle",
  "North Carolina",
] as const;
export type TargetMarket = (typeof TARGET_MARKETS)[number];

export const ROLE_CATEGORIES = [
  "Software Engineering",
  "Data Science",
  "Data Analytics",
  "Machine Learning / AI",
  "Fintech / Financial Services",
  "Risk Management / Risk Technology",
  "Quantitative / Trading Technology",
  "Product Analytics",
  "Cloud / DevOps / Infrastructure",
  "Cybersecurity",
  "General Technology Rotational Programs",
  "Other Entry-Level Technical Roles",
] as const;
export type RoleCategory = (typeof ROLE_CATEGORIES)[number];

export const DEFAULT_CATEGORIES: RoleCategory[] = [
  "Software Engineering",
  "Data Science",
  "Data Analytics",
  "Fintech / Financial Services",
  "Risk Management / Risk Technology",
  "General Technology Rotational Programs",
];

export type WorkArrangement = "onsite" | "hybrid" | "remote" | "unknown";
export type SeniorityLevel = "new_grad" | "entry_level" | "early_career" | "unknown";

// What an adapter hands back before normalization/scoring.
export type RawJobListing = {
  externalId?: string;
  source: string;
  sourceUrl: string;
  applicationUrl: string;

  companyName: string;
  companyCareersUrl?: string;

  title: string;
  description?: string;
  employmentType?: string;

  rawLocation?: string;
  postedAt?: string; // ISO date, if the source reports one
};

export type SearchParams = {
  locations: TargetMarket[];
  categories: RoleCategory[];
  keywords: string[];
  excludedKeywords: string[];
  priorityCompanies: string[];
  includeRemote: boolean;
  includeHybrid: boolean;
  includeInternships: boolean;
  includeContractRoles: boolean;
  maxExperienceYears: number;
  forceRefresh?: boolean;
};

export type SourceQueryResult = {
  source: string;
  status: "success" | "partial" | "unavailable" | "error";
  jobsFound: number;
  message?: string;
};

export type NormalizedJob = {
  externalId?: string;
  source: string;
  sourceUrl: string;
  applicationUrl: string;
  urlVerifiedAt?: string;

  companyName: string;
  companyCareersUrl?: string;

  title: string;
  normalizedTitle: string;
  description?: string;
  employmentType?: string;
  workArrangement: WorkArrangement;

  rawLocation?: string;
  city?: string;
  state?: string;
  region?: TargetMarket;
  country?: string;

  roleCategories: RoleCategory[];
  seniorityLevel: SeniorityLevel;

  postedAt?: string;
  retrievedAt: string;

  matchScore: number;
  matchReasons: string[];

  isPriorityCompany: boolean;
  isFintechRelevant: boolean;
  isRiskRelevant: boolean;

  fingerprint: string;
  isExcluded: boolean;
  exclusionReason?: string;
};

export type CompanySuggestion = {
  name: string;
  industry?: string;
  regions: TargetMarket[];
  reason: string;
  hasNewGradPipeline: boolean;
  careersUrl?: string;
  isUserPriority: boolean;
};
