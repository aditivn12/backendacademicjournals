import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db/prisma";
import { savedJobUpdateSchema } from "@/lib/validation";

export async function PATCH(req: NextRequest, { params }: { params: { id: string } }) {
  const body = await req.json().catch(() => null);
  const parsed = savedJobUpdateSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "Invalid request", details: parsed.error.flatten() }, { status: 400 });
  }

  try {
    const savedJob = await prisma.savedJob.update({
      where: { id: params.id },
      data: parsed.data,
      include: { jobListing: true },
    });
    return NextResponse.json({ savedJob });
  } catch (err) {
    console.error("Update saved job failed:", err);
    return NextResponse.json({ error: "Saved job not found or update failed" }, { status: 404 });
  }
}

export async function DELETE(_req: NextRequest, { params }: { params: { id: string } }) {
  try {
    await prisma.savedJob.delete({ where: { id: params.id } });
    return NextResponse.json({ deleted: true });
  } catch (err) {
    console.error("Delete saved job failed:", err);
    return NextResponse.json({ error: "Saved job not found" }, { status: 404 });
  }
}
