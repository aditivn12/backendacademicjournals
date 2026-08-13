"use client";

import { useEffect, useMemo, useState } from "react";

type SavedJobRow = {
  id: string;
  status: string;
  notes: string | null;
  savedAt: string;
  jobListing: {
    id: string;
    companyName: string;
    title: string;
    city: string | null;
    rawLocation: string | null;
    applicationUrl: string;
    source: string;
  };
};

const STATUS_OPTIONS = ["saved", "applied", "interviewing", "rejected", "offer", "not_interested"];

export function SavedJobsTable() {
  const [rows, setRows] = useState<SavedJobRow[]>([]);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    const res = await fetch("/api/saved-jobs");
    const data = await res.json();
    setRows(data.savedJobs ?? []);
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  async function updateRow(id: string, patch: Partial<{ status: string; notes: string }>) {
    setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
    await fetch(`/api/saved-jobs/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
  }

  async function removeRow(id: string) {
    setRows((prev) => prev.filter((r) => r.id !== id));
    await fetch(`/api/saved-jobs/${id}`, { method: "DELETE" });
  }

  const filtered = useMemo(
    () => (statusFilter === "all" ? rows : rows.filter((r) => r.status === statusFilter)),
    [rows, statusFilter],
  );

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="text-xl font-medium">Saved jobs</h1>
        <a href="/" className="text-sm text-accent font-medium">
          ← Back to search
        </a>
      </div>

      <div className="flex items-center gap-2 mb-4 text-xs">
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="border border-line rounded-md px-2 py-1"
        >
          <option value="all">All statuses</option>
          {STATUS_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <a href="/api/jobs/export" className="border border-line rounded-md px-2.5 py-1 ml-auto">
          Export to CSV ↓
        </a>
      </div>

      {loading && <p className="text-sm text-ink/50">Loading…</p>}
      {!loading && filtered.length === 0 && (
        <p className="text-sm text-ink/50 py-10 text-center">No saved jobs yet.</p>
      )}

      <div className="flex flex-col gap-3">
        {filtered.map((row) => (
          <div key={row.id} className="rounded-xl bg-white border border-line p-4">
            <div className="flex justify-between items-start gap-3">
              <div>
                <p className="font-medium text-sm">{row.jobListing.title}</p>
                <p className="text-xs text-ink/60 mt-0.5">
                  {row.jobListing.companyName} · {row.jobListing.city ?? row.jobListing.rawLocation ?? ""}
                </p>
              </div>
              <select
                value={row.status}
                onChange={(e) => updateRow(row.id, { status: e.target.value })}
                className="text-xs border border-line rounded-md px-2 py-1"
              >
                {STATUS_OPTIONS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <textarea
              className="w-full text-xs border border-line rounded-md p-2 mt-3"
              rows={2}
              placeholder="Notes"
              defaultValue={row.notes ?? ""}
              onBlur={(e) => updateRow(row.id, { notes: e.target.value })}
            />
            <div className="flex items-center gap-2 text-xs mt-2">
              <a
                href={row.jobListing.applicationUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="border border-line rounded-md px-3 py-1.5 font-medium"
              >
                View / apply ↗
              </a>
              <button
                type="button"
                onClick={() => removeRow(row.id)}
                className="border border-line rounded-md px-3 py-1.5"
              >
                Remove
              </button>
              <span className="text-ink/40 ml-auto">
                Saved {new Date(row.savedAt).toLocaleDateString()}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
