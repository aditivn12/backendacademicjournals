import { NextRequest, NextResponse } from "next/server";
import { recommendCompanies } from "@/lib/companies/recommendCompanies";
import { prisma } from "@/lib/db/prisma";
import { isRateLimited } from "@/lib/rateLimit";
import { runSearch } from "@/lib/search";
import { searchParamsSchema } from "@/lib/validation";

export async function POST(req: NextRequest) {
  if (isRateLimited()) {
    return NextResponse.json({ error: "Too many searches. Wait a moment and try again." }, { status: 429 });
  }

  const body = await req.json().catch(() => null);
  const parsed = searchParamsSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid search parameters", details: parsed.error.flatten() }, { status: 400 });
  }

  const params = parsed.data;

  try {
    const result = await runSearch(params);

    // Persist each job so it can later be saved/tracked. Upsert on
    // fingerprint so re-running the same search doesn't create duplicate rows.
    const persisted = await Promise.all(
      result.jobs.map(async (job) => {
        const row = await prisma.jobListing.upsert({
          where: { fingerprint: job.fingerprint },
          update: {
            matchScore: job.matchScore,
            matchReasons: JSON.stringify(job.matchReasons),
            isActive: true,
            updatedAt: new Date(),
          },
          create: {
            externalId: job.externalId,
            source: job.source,
            sourceUrl: job.sourceUrl,
            applicationUrl: job.applicationUrl,
            urlVerifiedAt: job.urlVerifiedAt ? new Date(job.urlVerifiedAt) : undefined,
            companyName: job.companyName,
            companyCareersUrl: job.companyCareersUrl,
            title: job.title,
            normalizedTitle: job.normalizedTitle,
            description: job.description,
            employmentType: job.employmentType,
            workArrangement: job.workArrangement,
            rawLocation: job.rawLocation,
            city: job.city,
            state: job.state,
            region: job.region,
            country: job.country,
            roleCategories: JSON.stringify(job.roleCategories),
            seniorityLevel: job.seniorityLevel,
            postedAt: job.postedAt ? new Date(job.postedAt) : undefined,
            retrievedAt: new Date(job.retrievedAt),
            matchScore: job.matchScore,
            matchReasons: JSON.stringify(job.matchReasons),
            isPriorityCompany: job.isPriorityCompany,
            isFintechRelevant: job.isFintechRelevant,
            isRiskRelevant: job.isRiskRelevant,
            fingerprint: job.fingerprint,
            isActive: true,
          },
        });
        return { ...job, id: row.id };
      }),
    );

    const suggestedCompanies = recommendCompanies(
      params.locations,
      params.categories,
      params.priorityCompanies,
    );

    await prisma.searchRun.create({
      data: {
        paramsJson: JSON.stringify(params),
        resultCount: result.resultCount,
        sourcesQueried: JSON.stringify(result.sourcesQueried),
        warnings: JSON.stringify(result.warnings),
      },
    });

    return NextResponse.json({
      searchedAt: result.searchedAt,
      resultCount: result.resultCount,
      jobs: persisted,
      suggestedCompanies,
      sourcesQueried: result.sourcesQueried,
      warnings: result.warnings,
      gapCompanies: result.gapCompanies,
    });
  } catch (err) {
    // Never let an unhandled exception fall through to an empty/non-JSON
    // response - the client always expects JSON back, success or failure.
    console.error("Search failed:", err);
    const message = err instanceof Error ? err.message : "Unknown server error";
    return NextResponse.json(
      {
        error:
          "Search failed on the server. If this is your first run, make sure you've run `npx prisma db push` to create the database.",
        detail: message,
      },
      { status: 500 },
    );
  }
}
