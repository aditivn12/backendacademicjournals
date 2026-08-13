"use client";

import { SourceQueryResult } from "@/components/types";

type Props = {
  loading: boolean;
  resultCount: number;
  lastRefreshed?: string;
  sourcesQueried: SourceQueryResult[];
  warnings: string[];
  onRefresh: () => void;
};

export function SearchProgress({ loading, resultCount, lastRefreshed, sourcesQueried, warnings, onRefresh }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs text-ink/60">
        <span>
          {loading
            ? "Searching…"
            : lastRefreshed
              ? `${resultCount} results · last refreshed ${new Date(lastRefreshed).toLocaleTimeString()}`
              : "No search run yet"}
        </span>
        {lastRefreshed && !loading && (
          <button type="button" onClick={onRefresh} className="text-accent font-medium">
            Refresh results
          </button>
        )}
      </div>
      {sourcesQueried.length > 0 && (
        <div className="flex flex-wrap gap-1.5 text-[11px] text-ink/50">
          {sourcesQueried.map((s) => (
            <span key={s.source}>
              {s.source}: {s.status === "success" ? `${s.jobsFound} found` : s.status}
            </span>
          ))}
        </div>
      )}
      {warnings.map((w, i) => (
        <p key={i} className="text-[11px] text-warn bg-warnSoft px-2 py-1 rounded-md">
          {w}
        </p>
      ))}
    </div>
  );
}
