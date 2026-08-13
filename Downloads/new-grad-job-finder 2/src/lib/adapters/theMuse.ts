import { JobSourceAdapter } from "@/lib/adapters/types";
import { RawJobListing, RoleCategory, SearchParams, TargetMarket } from "@/lib/types";

// The Muse's public jobs API - no API key required for normal personal
// -use call volume. It already supports an entry-level filter and a
// category taxonomy that maps closely to our own role categories, which
// covers a lot of the classification work for us.
const THE_MUSE_LOCATION: Record<TargetMarket, string> = {
  "New York City": "New York, NY",
  "San Francisco Bay Area": "San Francisco, CA",
  Chicago: "Chicago, IL",
  Seattle: "Seattle, WA",
  "North Carolina": "Raleigh, NC",
};

// The Muse's own category strings - only include ones with a clean match;
// anything else is left to our own downstream classification.
const THE_MUSE_CATEGORY: Partial<Record<RoleCategory, string>> = {
  "Software Engineering": "Software Engineering",
  "Data Science": "Data Science",
  "Data Analytics": "Data and Analytics",
  "Machine Learning / AI": "Data Science",
  "Product Analytics": "Data and Analytics",
  "Cloud / DevOps / Infrastructure": "IT",
  Cybersecurity: "IT",
};

type MuseJob = {
  id: number;
  name: string;
  company?: { name?: string };
  locations?: { name?: string }[];
  refs?: { landing_page?: string };
  publication_date?: string;
  contents?: string;
};

export class TheMuseAdapter implements JobSourceAdapter {
  sourceName = "The Muse";

  async isAvailable(): Promise<boolean> {
    return true;
  }

  async searchJobs(params: SearchParams): Promise<RawJobListing[]> {
    const locations = params.locations.length > 0 ? params.locations : (Object.keys(THE_MUSE_LOCATION) as TargetMarket[]);
    const categories =
      params.categories.length > 0
        ? Array.from(new Set(params.categories.map((c) => THE_MUSE_CATEGORY[c]).filter(Boolean)))
        : [undefined];

    const results: RawJobListing[] = [];

    await Promise.all(
      locations.flatMap((market) =>
        (categories.length > 0 ? categories : [undefined]).map(async (category) => {
          const url = new URL("https://www.themuse.com/api/public/jobs");
          url.searchParams.set("page", "0");
          url.searchParams.set("location", THE_MUSE_LOCATION[market]);
          url.searchParams.set("level", "Entry Level");
          if (category) url.searchParams.set("category", category as string);

          try {
            const res = await fetch(url.toString(), { cache: "no-store" });
            if (!res.ok) return;
            const data = (await res.json()) as { results: MuseJob[] };

            for (const job of data.results ?? []) {
              results.push({
                externalId: String(job.id),
                source: this.sourceName,
                sourceUrl: job.refs?.landing_page ?? "",
                applicationUrl: job.refs?.landing_page ?? "",
                companyName: job.company?.name ?? "Unknown company",
                title: job.name,
                description: job.contents,
                rawLocation: job.locations?.[0]?.name,
                postedAt: job.publication_date,
              });
            }
          } catch {
            // One location/category combo failing shouldn't fail the adapter.
          }
        }),
      ),
    );

    return results.filter((j) => j.applicationUrl);
  }
}
