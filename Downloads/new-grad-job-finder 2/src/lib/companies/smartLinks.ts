import { SEEDED_COMPANIES } from "@/lib/companies/seededCompanies";
import { RoleCategory, TargetMarket } from "@/lib/types";

export type SmartLinks = {
  companyName: string;
  careersUrl?: string;
  googleSearchUrl: string;
  claudeInChromePrompt: string;
};

export function buildSmartLinks(
  companyName: string,
  categories: RoleCategory[],
  locations: TargetMarket[],
): SmartLinks {
  const seedMatch = SEEDED_COMPANIES.find(
    (c) => c.name.toLowerCase() === companyName.trim().toLowerCase(),
  );
  const careersUrl = seedMatch?.careersUrl;

  const categoryText = categories.length > 0 ? categories.slice(0, 2).join(" or ") : "new grad";
  const locationText = locations.length > 0 ? locations.join(" or ") : "";

  const googleSearchUrl = `https://www.google.com/search?q=${encodeURIComponent(
    `${companyName} new grad ${categoryText} jobs ${locationText}`,
  )}`;

  const claudeInChromePrompt = `Go to ${
    careersUrl ? careersUrl : `${companyName}'s careers site`
  } and search for new-grad or entry-level roles${
    categories.length > 0 ? ` in ${categories.join(", ")}` : ""
  }${
    locations.length > 0 ? ` located in ${locations.join(", ")}` : ""
  }. List each opening with its title, location, and a direct link to apply.`;

  return { companyName, careersUrl, googleSearchUrl, claudeInChromePrompt };
}
