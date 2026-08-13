"use client";

import { DEFAULT_CATEGORIES, ROLE_CATEGORIES, RoleCategory, TARGET_MARKETS, TargetMarket } from "@/lib/types";
import { SearchFormState } from "@/components/types";

type Props = {
  form: SearchFormState;
  onChange: (next: SearchFormState) => void;
  onSearch: () => void;
  onReset: () => void;
  loading: boolean;
};

export const DEFAULT_PRIORITY_COMPANIES = "JPMorgan Chase\nSalesforce\nCitibank";

export function defaultSearchForm(): SearchFormState {
  return {
    locations: ["New York City", "North Carolina"],
    categories: DEFAULT_CATEGORIES as RoleCategory[],
    keywords: [],
    excludedKeywords: [],
    priorityCompanies: DEFAULT_PRIORITY_COMPANIES.split("\n"),
    includeRemote: true,
    includeHybrid: true,
    includeInternships: false,
    includeContractRoles: false,
    maxExperienceYears: 2,
  };
}

function toggleInArray<T>(arr: T[], value: T): T[] {
  return arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value];
}

export function SearchFilters({ form, onChange, onSearch, onReset, loading }: Props) {
  return (
    <div className="flex flex-col gap-5 rounded-xl bg-white border border-line p-4 h-fit sticky top-4">
      <div>
        <p className="text-xs font-medium text-ink/60 mb-2">Target locations</p>
        <div className="flex flex-col gap-1.5">
          {TARGET_MARKETS.map((market) => (
            <label key={market} className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={form.locations.includes(market)}
                onChange={() => onChange({ ...form, locations: toggleInArray(form.locations, market) })}
              />
              {market}
            </label>
          ))}
        </div>
      </div>

      <div>
        <p className="text-xs font-medium text-ink/60 mb-2">Role categories</p>
        <div className="flex flex-wrap gap-1.5">
          {ROLE_CATEGORIES.map((category) => {
            const active = form.categories.includes(category);
            return (
              <button
                key={category}
                type="button"
                onClick={() => onChange({ ...form, categories: toggleInArray(form.categories, category) })}
                className={`text-xs px-2.5 py-1 rounded-md border ${
                  active ? "bg-accentSoft border-accent text-accent" : "border-line text-ink/70"
                }`}
              >
                {category}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <label className="text-xs font-medium text-ink/60 mb-2 block">
          Priority companies (optional)
        </label>
        <textarea
          rows={4}
          className="w-full text-sm border border-line rounded-md p-2"
          placeholder={"Wells Fargo\nCapital One"}
          value={form.priorityCompanies.join("\n")}
          onChange={(e) =>
            onChange({
              ...form,
              priorityCompanies: e.target.value
                .split(/[\n,]/)
                .map((c) => c.trim())
                .filter(Boolean),
            })
          }
        />
        <p className="text-[11px] text-ink/50 mt-1">
          One per line, or comma-separated. These get a ranking boost and a "priority company" label.
        </p>
      </div>

      <div>
        <label className="text-xs font-medium text-ink/60 mb-2 block">Keywords</label>
        <input
          type="text"
          className="w-full text-sm border border-line rounded-md p-2"
          placeholder="Python, SQL"
          value={form.keywords.join(", ")}
          onChange={(e) =>
            onChange({ ...form, keywords: e.target.value.split(",").map((k) => k.trim()).filter(Boolean) })
          }
        />
      </div>

      <div>
        <label className="text-xs font-medium text-ink/60 mb-2 block">Excluded keywords</label>
        <input
          type="text"
          className="w-full text-sm border border-line rounded-md p-2"
          placeholder="senior, manager"
          value={form.excludedKeywords.join(", ")}
          onChange={(e) =>
            onChange({
              ...form,
              excludedKeywords: e.target.value.split(",").map((k) => k.trim()).filter(Boolean),
            })
          }
        />
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.includeRemote}
            onChange={() => onChange({ ...form, includeRemote: !form.includeRemote })}
          />
          Include remote jobs
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.includeHybrid}
            onChange={() => onChange({ ...form, includeHybrid: !form.includeHybrid })}
          />
          Include hybrid jobs
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.includeInternships}
            onChange={() => onChange({ ...form, includeInternships: !form.includeInternships })}
          />
          Include internships
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={form.includeContractRoles}
            onChange={() => onChange({ ...form, includeContractRoles: !form.includeContractRoles })}
          />
          Include contract roles
        </label>
      </div>

      <div>
        <label className="text-xs font-medium text-ink/60 mb-2 block">
          Max years of experience: {form.maxExperienceYears}
        </label>
        <input
          type="range"
          min={0}
          max={5}
          step={1}
          value={form.maxExperienceYears}
          onChange={(e) => onChange({ ...form, maxExperienceYears: parseInt(e.target.value, 10) })}
          className="w-full"
        />
      </div>

      <div className="flex flex-col gap-2 pt-1">
        <button
          type="button"
          onClick={onSearch}
          disabled={loading}
          className="w-full rounded-md bg-accent text-white text-sm font-medium py-2 disabled:opacity-60"
        >
          {loading ? "Searching…" : "Search open roles"}
        </button>
        <button type="button" onClick={onReset} className="w-full rounded-md border border-line text-sm py-2">
          Reset filters
        </button>
      </div>
    </div>
  );
}
