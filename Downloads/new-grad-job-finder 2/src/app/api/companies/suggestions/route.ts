import { NextRequest, NextResponse } from "next/server";
import { recommendCompanies } from "@/lib/companies/recommendCompanies";
import { ROLE_CATEGORIES, RoleCategory, TARGET_MARKETS, TargetMarket } from "@/lib/types";

export async function GET(req: NextRequest) {
  const locationsParam = req.nextUrl.searchParams.get("locations");
  const categoriesParam = req.nextUrl.searchParams.get("categories");
  const priorityParam = req.nextUrl.searchParams.get("priorityCompanies");

  const locations = (locationsParam?.split(",").filter(Boolean) ?? []).filter((l) =>
    TARGET_MARKETS.includes(l as TargetMarket),
  ) as TargetMarket[];
  const categories = (categoriesParam?.split(",").filter(Boolean) ?? []).filter((c) =>
    ROLE_CATEGORIES.includes(c as RoleCategory),
  ) as RoleCategory[];
  const priorityCompanies = priorityParam?.split(",").filter(Boolean) ?? [];

  const suggestions = recommendCompanies(locations, categories, priorityCompanies);
  return NextResponse.json({ suggestedCompanies: suggestions });
}
