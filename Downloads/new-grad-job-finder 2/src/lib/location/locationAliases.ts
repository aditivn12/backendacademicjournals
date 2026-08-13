import { TargetMarket } from "@/lib/types";

// Lowercase alias -> target market. Used both to expand search queries and
// to resolve a raw "city, state" string from a job listing back to one of
// the 5 markets the user can filter on.
export const LOCATION_ALIASES: Record<TargetMarket, string[]> = {
  "New York City": [
    "new york",
    "new york, ny",
    "new york city",
    "nyc",
    "manhattan",
    "brooklyn",
    "jersey city",
    "hoboken",
  ],
  "San Francisco Bay Area": [
    "san francisco",
    "san francisco, ca",
    "san jose",
    "san jose, ca",
    "oakland",
    "oakland, ca",
    "palo alto",
    "mountain view",
    "sunnyvale",
    "menlo park",
    "bay area",
    "sf bay area",
  ],
  Chicago: ["chicago", "chicago, il", "greater chicago area"],
  Seattle: ["seattle", "seattle, wa", "bellevue", "redmond", "greater seattle area"],
  "North Carolina": [
    "raleigh",
    "durham",
    "cary",
    "apex",
    "research triangle",
    "north carolina",
    "charlotte",
    "rtp",
  ],
};

/** Query-friendly search terms per market, used for expanding a broad
 * category+location search into several focused queries. */
export const LOCATION_SEARCH_TERMS: Record<TargetMarket, string[]> = {
  "New York City": ["New York", "NYC", "Manhattan"],
  "San Francisco Bay Area": ["San Francisco", "San Jose", "Bay Area"],
  Chicago: ["Chicago"],
  Seattle: ["Seattle"],
  "North Carolina": ["Raleigh", "Charlotte", "Durham", "North Carolina"],
};
