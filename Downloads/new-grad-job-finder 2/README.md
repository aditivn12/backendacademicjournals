# New grad job finder

Personal-use web app for finding new-grad / entry-level roles in NYC, the SF
Bay Area, Chicago, Seattle, and North Carolina, with fintech/risk relevance
weighted for a Wells Fargo Risk Management internship background.

Default priority companies (edit anytime in the app): **JPMorgan Chase,
Salesforce, Citibank**.

## Setup

```bash
npm install
cp .env.example .env
npx prisma db push      # creates dev.db (SQLite)
npm run dev
```

Open http://localhost:3000.

> `npm install` and `prisma db push` need normal internet access to
> download packages and the Prisma query engine binary. If you're running
> this inside a network-restricted sandbox, that download will fail - it
> works fine on a normal machine or cloud VM.

## How search works now (revised architecture)

The first version of this app tried to integrate directly with each
company's ATS (Greenhouse, Lever, Ashby, Workday). That fell apart on
contact with reality: enterprise employers like JPMorgan, Citibank, and
Salesforce don't use those platforms, so covering them meant reverse
-engineering a different, fragile, per-company integration for each one -
exactly the one-at-a-time manual effort this app is supposed to remove.

The current design instead has two tiers:

**Tier 1/2 - broad aggregator search (automated, no per-company config)**

Two job-aggregator APIs do the actual searching:
- **Adzuna** (`src/lib/adapters/adzuna.ts`) - broad US job aggregator,
  needs a free API key from https://developer.adzuna.com/. Set
  `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in `.env`.
- **The Muse** (`src/lib/adapters/theMuse.ts`) - skews toward tech/media
  employers, has a built-in "Entry Level" filter, no API key needed.

Both cover thousands of companies automatically. Results are normalized,
scored, and deduplicated exactly as before.

**Tier 3 - gap detection + hand-off to Claude in Chrome**

After each search, the app diffs your priority-company list against which
companies actually showed up in the aggregator results
(`src/lib/companies/companyCoverage.ts`). Anything with zero matches shows
up in a "Not covered by automated search" panel with:
- a link to the company's careers page (if known),
- a one-click Google search link pre-filled with role + location,
- a ready-to-copy prompt for Claude in Chrome, e.g. "Go to
  careers.jpmorgan.com and search for new-grad or entry-level roles in
  Risk Management / Risk Technology located in New York City. List each
  opening with its title, location, and a direct link to apply."

This means large/enterprise employers that aggregators don't reliably
index (which, in practice, is most banks) get a fast, low-friction manual
check instead of a maintenance-heavy scraper that breaks the moment the
company redesigns their careers page.

## Two ways to run it

**Mock data** (no live network calls, works immediately): set
`USE_MOCK_DATA="true"` in `.env`. Serves 30+ seeded sample jobs including
JPMorgan Chase, Salesforce, and Citibank test listings, a duplicate pair
(dedup check), an excluded senior role, and a remote role tied to a
target-region employer. Good for verifying the UI/scoring before wiring
up a live Adzuna key.

**Live data**: set `USE_MOCK_DATA="false"` (default) and add your Adzuna
credentials to `.env`. The Muse needs no setup.

## Adding a new aggregator source

1. Create `src/lib/adapters/yourSource.ts` implementing `JobSourceAdapter`
   (see `src/lib/adapters/types.ts`) - return `RawJobListing[]`.
2. Register it in `getAdapters()` in `src/lib/search.ts`.

## Adding or editing suggested companies

Edit `src/lib/companies/seededCompanies.ts` directly - a plain TypeScript
array. No admin UI for this personal-use version; the file is the config.

## Notes on what was simplified for personal use

- SQLite instead of Postgres, no BullMQ/Redis, no multi-user auth.
- Dedup uses exact-match tiers (URL, then source+ID, then
  company+title+location) rather than fuzzy text matching.
- Roles matching senior/staff/lead/manager terms are hard-excluded rather
  than just scored down, so they don't clutter the results list.
- Rate limiting and caching are in-memory (reset on server restart) -
  fine for single-user local use.
- Gap-company coverage is checked against your priority-company list
  only, not the full suggested-companies dataset, to keep the "needs a
  manual check" list short and actionable.

## Project structure

```
src/
  app/
    page.tsx                  main search page
    saved/page.tsx             saved jobs page
    api/jobs/search             POST - run a search
    api/jobs/export              GET  - CSV export of saved jobs
    api/saved-jobs                GET/POST list+create; [id] for PATCH/DELETE
    api/companies/suggestions      GET - standalone suggested-companies lookup
  components/                  SearchFilters, JobCard, GapCompanies, JobFinderApp, etc.
  lib/
    adapters/                   adzuna, theMuse, mock
    ranking/                    scoreJob, classifyJob, deduplicateJobs
    location/                   locationAliases, normalizeLocation
    companies/                  seededCompanies, recommendCompanies, companyCoverage, smartLinks
    search.ts                   orchestrates adapters -> normalize -> filter -> gap detection -> cache
    validation.ts               Zod schemas for API inputs
prisma/schema.prisma
```
