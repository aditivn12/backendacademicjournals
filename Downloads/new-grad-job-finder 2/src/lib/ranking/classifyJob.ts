import { RoleCategory, SeniorityLevel } from "@/lib/types";

// Keyword -> category map. A job can match multiple categories.
const CATEGORY_KEYWORDS: Record<RoleCategory, string[]> = {
  "Software Engineering": ["software engineer", "swe", "developer", "engineer i", "engineer, university"],
  "Data Science": ["data scientist", "data science"],
  "Data Analytics": ["data analyst", "business intelligence", "bi analyst"],
  "Machine Learning / AI": ["machine learning", "ml engineer", "ai engineer", "artificial intelligence"],
  "Fintech / Financial Services": [
    "fintech",
    "payments",
    "financial services",
    "bank",
    "banking",
    "credit",
    "lending",
  ],
  "Risk Management / Risk Technology": ["risk analyst", "risk technology", "risk management", "credit risk", "fraud"],
  "Quantitative / Trading Technology": ["quantitative analyst", "quant", "trading technology", "trading"],
  "Product Analytics": ["product analyst", "product analytics"],
  "Cloud / DevOps / Infrastructure": ["cloud engineer", "devops", "infrastructure engineer", "site reliability"],
  Cybersecurity: ["cybersecurity", "security analyst", "security engineer"],
  "General Technology Rotational Programs": ["rotational", "technology analyst program", "technology rotational"],
  "Other Entry-Level Technical Roles": [],
};

const NEW_GRAD_TERMS = ["new grad", "university graduate", "campus", "early career", "university recruiting"];
const ENTRY_LEVEL_TERMS = ["entry level", "entry-level", "associate", "analyst", "junior", "engineer i", "engineer, entry"];
const SENIORITY_EXCLUDE_TERMS = [
  "senior",
  "staff",
  "principal",
  " lead ",
  "manager",
  "director",
  "vice president",
  "vp,",
  "vp ",
  "executive",
];

export function classifyRoleCategories(title: string, description?: string): RoleCategory[] {
  const haystack = `${title} ${description ?? ""}`.toLowerCase();
  const matches: RoleCategory[] = [];

  for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS) as [
    RoleCategory,
    string[],
  ][]) {
    if (keywords.some((kw) => haystack.includes(kw))) {
      matches.push(category);
    }
  }

  if (matches.length === 0) matches.push("Other Entry-Level Technical Roles");
  return matches;
}

export function classifySeniority(title: string, description?: string): SeniorityLevel {
  const haystack = `${title} ${description ?? ""}`.toLowerCase();

  if (NEW_GRAD_TERMS.some((t) => haystack.includes(t))) return "new_grad";
  if (ENTRY_LEVEL_TERMS.some((t) => haystack.includes(t))) return "entry_level";
  if (haystack.includes("early career")) return "early_career";
  return "unknown";
}

export function hasSeniorityExclusionTerms(title: string): boolean {
  const lower = ` ${title.toLowerCase()} `;
  return SENIORITY_EXCLUDE_TERMS.some((t) => lower.includes(t));
}

/** Best-effort extraction of a required years-of-experience number from
 * free text, e.g. "requires 5+ years of experience" -> 5. Returns
 * undefined if no such pattern is found (does not penalize on absence). */
export function extractMaxExperienceYears(description?: string): number | undefined {
  if (!description) return undefined;
  const match = description.match(/(\d{1,2})\s*\+?\s*years?/i);
  if (!match) return undefined;
  return parseInt(match[1], 10);
}

export function isFintechRelevant(title: string, description?: string): boolean {
  return CATEGORY_KEYWORDS["Fintech / Financial Services"].some((kw) =>
    `${title} ${description ?? ""}`.toLowerCase().includes(kw),
  );
}

export function isRiskRelevant(title: string, description?: string): boolean {
  return CATEGORY_KEYWORDS["Risk Management / Risk Technology"].some((kw) =>
    `${title} ${description ?? ""}`.toLowerCase().includes(kw),
  );
}
