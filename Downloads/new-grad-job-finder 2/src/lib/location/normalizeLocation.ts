import { LOCATION_ALIASES } from "@/lib/location/locationAliases";
import { TargetMarket, WorkArrangement } from "@/lib/types";

export type NormalizedLocation = {
  region?: TargetMarket;
  city?: string;
  state?: string;
  rawLocation?: string;
};

/**
 * Resolve a raw location string (as reported by a source) to one of the 5
 * target markets, if it matches. If nothing matches, region is left
 * undefined and the raw text is preserved verbatim per spec section 2.2:
 * "If exact city information is unavailable, show the listing's raw
 * location text."
 */
export function normalizeLocation(rawLocation: string | undefined): NormalizedLocation {
  if (!rawLocation) return {};

  const lower = rawLocation.toLowerCase();

  for (const [market, aliases] of Object.entries(LOCATION_ALIASES) as [
    TargetMarket,
    string[],
  ][]) {
    if (aliases.some((alias) => lower.includes(alias))) {
      const parts = rawLocation.split(",").map((p) => p.trim());
      return {
        region: market,
        city: parts[0],
        state: parts.length > 1 ? parts[1] : undefined,
        rawLocation,
      };
    }
  }

  return { rawLocation };
}

const REMOTE_TERMS = ["remote"];
const HYBRID_TERMS = ["hybrid"];
const ONSITE_TERMS = ["onsite", "on-site", "in office", "in-office"];

/** Infer work arrangement from raw location + description text. Defaults
 * to "unknown" rather than guessing, per spec section 6 data model. */
export function inferWorkArrangement(
  rawLocation?: string,
  description?: string,
): WorkArrangement {
  const haystack = `${rawLocation ?? ""} ${description ?? ""}`.toLowerCase();

  if (REMOTE_TERMS.some((t) => haystack.includes(t))) return "remote";
  if (HYBRID_TERMS.some((t) => haystack.includes(t))) return "hybrid";
  if (ONSITE_TERMS.some((t) => haystack.includes(t))) return "onsite";
  return "unknown";
}
