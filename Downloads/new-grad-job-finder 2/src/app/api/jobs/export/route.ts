import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";

function csvEscape(value: string): string {
  if (value.includes(",") || value.includes('"') || value.includes("\n")) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

export async function GET(req: NextRequest) {
  const status = req.nextUrl.searchParams.get("status");

  try {
    const savedJobs = await prisma.savedJob.findMany({
      where: status ? { status } : undefined,
      include: { jobListing: true },
      orderBy: { savedAt: "desc" },
    });

    const header = [
      "Company",
      "Title",
      "Location",
      "Work Arrangement",
      "Category",
      "Posted Date",
      "Application URL",
      "Source",
      "Status",
      "Notes",
      "Date Saved",
    ];

    const rows = savedJobs.map((saved: (typeof savedJobs)[number]) => {
      const job = saved.jobListing;
      const categories = JSON.parse(job.roleCategories ?? "[]").join("; ");
      return [
        job.companyName,
        job.title,
        job.city ?? job.rawLocation ?? "",
        job.workArrangement ?? "",
        categories,
        job.postedAt ? job.postedAt.toISOString().slice(0, 10) : "",
        job.applicationUrl,
        job.source,
        saved.status,
        saved.notes ?? "",
        saved.savedAt.toISOString().slice(0, 10),
      ]
        .map((v) => csvEscape(String(v)))
        .join(",");
    });

    const csv = [header.join(","), ...rows].join("\n");

    return new NextResponse(csv, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": `attachment; filename="saved-jobs-${new Date().toISOString().slice(0, 10)}.csv"`,
      },
    });
  } catch (err) {
    console.error("Export failed:", err);
    return NextResponse.json(
      { error: "Export failed. Make sure the database has been created with `npx prisma db push`." },
      { status: 500 },
    );
  }
}
