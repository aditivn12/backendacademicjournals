"use client";

import { JobResult } from "@/components/types";

type Props = {
  job: JobResult;
  onSave: (job: JobResult) => void;
  onHide: (job: JobResult) => void;
  onMarkApplied: (job: JobResult) => void;
  isSaved: boolean;
  isHidden: boolean;
};

function scoreColor(score: number) {
  if (score >= 75) return "bg-accentSoft text-accent";
  if (score >= 50) return "bg-warnSoft text-warn";
  return "bg-line text-ink/60";
}

export function JobCard({ job, onSave, onHide, onMarkApplied, isSaved, isHidden }: Props) {
  if (isHidden) return null;

  return (
    <div className="rounded-xl bg-white border border-line p-4">
      <div className="flex justify-between items-start gap-3">
        <div>
          <p className="font-medium text-sm">{job.title}</p>
          <p className="text-xs text-ink/60 mt-0.5">
            {job.companyName} · {job.city ?? job.rawLocation ?? "Location not provided"}
            {job.workArrangement !== "unknown" ? ` · ${job.workArrangement}` : ""}
          </p>
        </div>
        <span className={`text-sm font-medium px-2.5 py-1 rounded-md shrink-0 ${scoreColor(job.matchScore)}`}>
          {job.matchScore}
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5 my-2">
        {job.roleCategories.map((c) => (
          <span key={c} className="text-[11px] bg-accentSoft text-accent px-2 py-0.5 rounded-md">
            {c}
          </span>
        ))}
        {job.isPriorityCompany && (
          <span className="text-[11px] bg-warnSoft text-warn px-2 py-0.5 rounded-md">Priority company</span>
        )}
      </div>

      <p className="text-xs text-ink/60 mb-3">
        {job.matchReasons.length > 0 ? job.matchReasons.join(", ") : "Match details not available"}
        {job.postedAt ? "" : " · Date not provided"}
      </p>

      <div className="flex items-center gap-2 text-xs">
        <a
          href={job.applicationUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="border border-line rounded-md px-3 py-1.5 font-medium"
        >
          View / apply on source site ↗
        </a>
        <button
          type="button"
          onClick={() => onSave(job)}
          className={`border border-line rounded-md px-3 py-1.5 ${isSaved ? "bg-accentSoft text-accent" : ""}`}
        >
          {isSaved ? "Saved" : "Save"}
        </button>
        <button type="button" onClick={() => onMarkApplied(job)} className="border border-line rounded-md px-3 py-1.5">
          Mark applied
        </button>
        <button type="button" onClick={() => onHide(job)} className="border border-line rounded-md px-3 py-1.5">
          Hide
        </button>
        <span className="text-ink/40 ml-auto">{job.source}</span>
      </div>
    </div>
  );
}
