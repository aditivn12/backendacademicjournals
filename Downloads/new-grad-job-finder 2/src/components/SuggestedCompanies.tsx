"use client";

import { CompanySuggestion } from "@/components/types";

export function SuggestedCompanies({ companies }: { companies: CompanySuggestion[] }) {
  if (companies.length === 0) return null;

  return (
    <div className="rounded-xl bg-white border border-line p-4">
      <p className="text-xs font-medium text-ink/60 mb-3">Suggested companies to explore</p>
      <div className="grid grid-cols-2 gap-2">
        {companies.map((company) => (
          <div key={company.name} className="border border-line rounded-md p-2.5">
            <div className="flex justify-between items-start gap-2">
              <p className="text-sm font-medium">{company.name}</p>
              {company.hasNewGradPipeline && (
                <span className="text-[10px] bg-accentSoft text-accent px-1.5 py-0.5 rounded-md shrink-0">
                  new-grad pipeline
                </span>
              )}
            </div>
            <p className="text-[11px] text-ink/50 mt-0.5">{company.industry}</p>
            <p className="text-xs text-ink/70 mt-1.5">{company.reason}</p>
            {company.careersUrl && (
              <a
                href={company.careersUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-accent inline-block mt-1.5"
              >
                Careers page ↗
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
