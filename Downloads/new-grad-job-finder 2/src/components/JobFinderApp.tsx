"use client";

import { useEffect, useMemo, useState } from "react";
import { defaultSearchForm, SearchFilters } from "@/components/SearchFilters";
import { SearchProgress } from "@/components/SearchProgress";
import { SuggestedCompanies } from "@/components/SuggestedCompanies";
import { GapCompanies } from "@/components/GapCompanies";
import { JobCard } from "@/components/JobCard";
import { CompanySuggestion, JobResult, SearchFormState, SourceQueryResult } from "@/components/types";

type SortOption = "best_match" | "recent" | "company" | "location" | "fintech_risk";

const HIDDEN_JOBS_KEY = "ngjf_hidden_job_ids";

export function JobFinderApp() {
  const [form, setForm] = useState<SearchFormState>(defaultSearchForm());
  const [loading, setLoading] = useState(false);
  const [jobs, setJobs] = useState<JobResult[]>([]);
  const [suggestedCompanies, setSuggestedCompanies] = useState<CompanySuggestion[]>([]);
  const [gapCompanies, setGapCompanies] = useState<
    { companyName: string; careersUrl?: string; googleSearchUrl: string; claudeInChromePrompt: string }[]
  >([]);
  const [sourcesQueried, setSourcesQueried] = useState<SourceQueryResult[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [lastRefreshed, setLastRefreshed] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();
  const [savedJobIds, setSavedJobIds] = useState<Set<string>>(new Set());
  const [hiddenJobIds, setHiddenJobIds] = useState<Set<string>>(new Set());

  const [sort, setSort] = useState<SortOption>("best_match");
  const [priorityOnly, setPriorityOnly] = useState(false);
  const [fintechOnly, setFintechOnly] = useState(false);
  const [riskOnly, setRiskOnly] = useState(false);
  const [remoteOnly, setRemoteOnly] = useState(false);
  const [recentOnly, setRecentOnly] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem(HIDDEN_JOBS_KEY);
    if (stored) setHiddenJobIds(new Set(JSON.parse(stored)));
  }, []);

  function persistHidden(next: Set<string>) {
    setHiddenJobIds(next);
    window.localStorage.setItem(HIDDEN_JOBS_KEY, JSON.stringify(Array.from(next)));
  }

  async function search(forceRefresh = false) {
    setLoading(true);
    setError(undefined);
    try {
      const res = await fetch("/api/jobs/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, forceRefresh }),
      });

      const text = await res.text();
      let data: any;
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        throw new Error(
          `Server returned an unreadable response (status ${res.status}). Check the terminal running "npm run dev" for the actual error.`,
        );
      }

      if (!res.ok) throw new Error(data.detail ?? data.error ?? "Search failed");

      setJobs(data.jobs);
      setSuggestedCompanies(data.suggestedCompanies ?? []);
      setGapCompanies(data.gapCompanies ?? []);
      setSourcesQueried(data.sourcesQueried ?? []);
      setWarnings(data.warnings ?? []);
      setLastRefreshed(data.searchedAt);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(job: JobResult) {
    setSavedJobIds((prev) => new Set(prev).add(job.id));
    await fetch("/api/saved-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobListingId: job.id, status: "saved" }),
    });
  }

  async function handleMarkApplied(job: JobResult) {
    setSavedJobIds((prev) => new Set(prev).add(job.id));
    await fetch("/api/saved-jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobListingId: job.id, status: "applied" }),
    });
  }

  function handleHide(job: JobResult) {
    persistHidden(new Set(hiddenJobIds).add(job.id));
  }

  const visibleJobs = useMemo(() => {
    let list = jobs.filter((j) => !hiddenJobIds.has(j.id));
    if (priorityOnly) list = list.filter((j) => j.isPriorityCompany);
    if (fintechOnly) list = list.filter((j) => j.isFintechRelevant);
    if (riskOnly) list = list.filter((j) => j.isRiskRelevant);
    if (remoteOnly) list = list.filter((j) => j.workArrangement === "remote");
    if (recentOnly) {
      list = list.filter((j) => {
        if (!j.postedAt) return false;
        const days = (Date.now() - new Date(j.postedAt).getTime()) / 86_400_000;
        return days <= 7;
      });
    }

    const sorted = [...list];
    if (sort === "recent") {
      sorted.sort((a, b) => new Date(b.postedAt ?? 0).getTime() - new Date(a.postedAt ?? 0).getTime());
    } else if (sort === "company") {
      sorted.sort((a, b) => a.companyName.localeCompare(b.companyName));
    } else if (sort === "location") {
      sorted.sort((a, b) => (a.region ?? "").localeCompare(b.region ?? ""));
    } else if (sort === "fintech_risk") {
      sorted.sort(
        (a, b) => Number(b.isFintechRelevant || b.isRiskRelevant) - Number(a.isFintechRelevant || a.isRiskRelevant),
      );
    } else {
      sorted.sort((a, b) => b.matchScore - a.matchScore);
    }
    return sorted;
  }, [jobs, hiddenJobIds, priorityOnly, fintechOnly, riskOnly, remoteOnly, recentOnly, sort]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <header className="mb-6">
        <div className="flex items-baseline justify-between">
          <h1 className="text-xl font-medium">New grad job finder</h1>
          <a href="/saved" className="text-sm text-accent font-medium">
            Saved jobs ↗
          </a>
        </div>
        <p className="text-sm text-ink/60 mt-1">
          Discover entry-level CS, data, fintech, and risk-tech roles.
        </p>
      </header>

      <div className="grid grid-cols-[260px_1fr] gap-5">
        <SearchFilters
          form={form}
          onChange={setForm}
          onSearch={() => search(false)}
          onReset={() => setForm(defaultSearchForm())}
          loading={loading}
        />

        <div className="flex flex-col gap-4">
          <SearchProgress
            loading={loading}
            resultCount={visibleJobs.length}
            lastRefreshed={lastRefreshed}
            sourcesQueried={sourcesQueried}
            warnings={warnings}
            onRefresh={() => search(true)}
          />

          {error && <p className="text-sm text-warn bg-warnSoft px-3 py-2 rounded-md">{error}</p>}

          <GapCompanies companies={gapCompanies} />

          <SuggestedCompanies companies={suggestedCompanies} />

          {jobs.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <select
                value={sort}
                onChange={(e) => setSort(e.target.value as SortOption)}
                className="border border-line rounded-md px-2 py-1"
              >
                <option value="best_match">Sort: best match</option>
                <option value="recent">Sort: most recently posted</option>
                <option value="company">Sort: company name</option>
                <option value="location">Sort: location</option>
                <option value="fintech_risk">Sort: fintech / risk relevance</option>
              </select>
              {[
                ["Priority companies only", priorityOnly, setPriorityOnly],
                ["Fintech only", fintechOnly, setFintechOnly],
                ["Risk-related only", riskOnly, setRiskOnly],
                ["Remote only", remoteOnly, setRemoteOnly],
                ["Posted in last 7 days", recentOnly, setRecentOnly],
              ].map(([label, value, setter]: any) => (
                <button
                  key={label}
                  type="button"
                  onClick={() => setter(!value)}
                  className={`border rounded-md px-2.5 py-1 ${
                    value ? "bg-accentSoft border-accent text-accent" : "border-line text-ink/70"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          <div className="flex flex-col gap-3">
            {visibleJobs.map((job) => (
              <JobCard
                key={job.id}
                job={job}
                onSave={handleSave}
                onHide={handleHide}
                onMarkApplied={handleMarkApplied}
                isSaved={savedJobIds.has(job.id)}
                isHidden={hiddenJobIds.has(job.id)}
              />
            ))}
            {!loading && jobs.length === 0 && !error && (
              <p className="text-sm text-ink/50 text-center py-10">
                Choose your locations and categories, then search open roles.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
