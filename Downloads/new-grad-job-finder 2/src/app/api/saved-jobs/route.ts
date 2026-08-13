import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";
import { savedJobCreateSchema } from "@/lib/validation";

function serverError(err: unknown) {
  console.error("saved-jobs route failed:", err);
  const message = err instanceof Error ? err.message : "Unknown server error";
  return NextResponse.json(
    {
      error:
        "Something went wrong on the server. If this is your first run, make sure you've run `npx prisma db push`.",
      detail: message,
    },
    { status: 500 },
  );
}

export async function GET() {
  try {
    const savedJobs = await prisma.savedJob.findMany({
      include: { jobListing: true },
      orderBy: { savedAt: "desc" },
    });
    return NextResponse.json({ savedJobs });
  } catch (err) {
    return serverError(err);
  }
}

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const parsed = savedJobCreateSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid request", details: parsed.error.flatten() }, { status: 400 });
  }

  const { jobListingId, status, notes } = parsed.data;

  try {
    const job = await prisma.jobListing.findUnique({ where: { id: jobListingId } });
    if (!job) {
      return NextResponse.json({ error: "Job listing not found" }, { status: 404 });
    }

    const savedJob = await prisma.savedJob.upsert({
      where: { jobListingId },
      update: { status, notes },
      create: { jobListingId, status, notes },
      include: { jobListing: true },
    });

    return NextResponse.json({ savedJob }, { status: 201 });
  } catch (err) {
    return serverError(err);
  }
}
