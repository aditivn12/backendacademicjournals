"use client";

import { useState } from "react";

type GapCompany = {
  companyName: string;
  careersUrl?: string;
  googleSearchUrl: string;
  claudeInChromePrompt: string;
};

export function GapCompanies({ companies }: { companies: GapCompany[] }) {
  const [copiedFor, setCopiedFor] = useState<string | null>(null);

  if (companies.length === 0) return null;

  async function copyPrompt(company: GapCompany) {
    await navigator.clipboard.writeText(company.claudeInChromePrompt);
    setCopiedFor(company.companyName);
    setTimeout(() => setCopiedFor(null), 2000);
  }

  return (
    <div className="rounded-xl bg-warnSoft border border-warn/30 p-4">
      <p className="text-xs font-medium text-warn mb-1">
        Not covered by automated search ({companies.length})
      </p>
      <p className="text-[11px] text-ink/60 mb-3">
        These priority companies didn't show up in aggregator results - likely too large/specific
        for broad job APIs to index reliably. Check them directly, or paste the prompt below into
        Claude in Chrome to have it check for you.
      </p>
      <div className="flex flex-col gap-2">
        {companies.map((company) => (
          <div key={company.companyName} className="bg-white border border-line rounded-md p-2.5">
            <div className="flex items-center justify-between gap-2">
              <p className="text-sm font-medium">{company.companyName}</p>
              <div className="flex gap-1.5 text-[11px]">
                {company.careersUrl && (
                  <a
                    href={company.careersUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="border border-line rounded-md px-2 py-1"
                  >
                    Careers page ↗
                  </a>
                )}
                <a
                  href={company.googleSearchUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="border border-line rounded-md px-2 py-1"
                >
                  Google search ↗
                </a>
                <button
                  type="button"
                  onClick={() => copyPrompt(company)}
                  className="border border-line rounded-md px-2 py-1"
                >
                  {copiedFor === company.companyName ? "Copied" : "Copy Claude in Chrome prompt"}
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
