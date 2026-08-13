import { RoleCategory, TargetMarket, WorkArrangement } from "@/lib/types";

export type JobResult = {
  id: string;
  companyName: string;
  title: string;
  rawLocation?: string;
  city?: string;
  region?: TargetMarket;
  workArrangement: WorkArrangement;
  roleCategories: RoleCategory[];
  matchScore: number;
  matchReasons: string[];
  postedAt?: string;
  source: string;
  applicationUrl: string;
  isPriorityCompany: boolean;
  isFintechRelevant: boolean;
  isRiskRelevant: boolean;
};

export type CompanySuggestion = {
  name: string;
  industry?: string;
  regions: TargetMarket[];
  reason: string;
  hasNewGradPipeline: boolean;
  careersUrl?: string;
};

export type SourceQueryResult = {
  source: string;
  status: "success" | "partial" | "unavailable" | "error";
  jobsFound: number;
  message?: string;
};

export type SearchFormState = {
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
};
