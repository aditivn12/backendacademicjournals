import { z } from "zod";
import { ROLE_CATEGORIES, TARGET_MARKETS } from "@/lib/types";

export const searchParamsSchema = z.object({
  locations: z.array(z.enum(TARGET_MARKETS)).default([]),
  categories: z.array(z.enum(ROLE_CATEGORIES)).default([]),
  keywords: z.array(z.string().max(60)).max(20).default([]),
  excludedKeywords: z.array(z.string().max(60)).max(20).default([]),
  priorityCompanies: z.array(z.string().max(120)).max(50).default([]),
  includeRemote: z.boolean().default(true),
  includeHybrid: z.boolean().default(true),
  includeInternships: z.boolean().default(false),
  includeContractRoles: z.boolean().default(false),
  maxExperienceYears: z.number().int().min(0).max(10).default(2),
  forceRefresh: z.boolean().optional().default(false),
});

export const savedJobCreateSchema = z.object({
  jobListingId: z.string().uuid(),
  status: z
    .enum(["saved", "applied", "interviewing", "rejected", "offer", "not_interested"])
    .default("saved"),
  notes: z.string().max(2000).optional(),
});

export const savedJobUpdateSchema = z.object({
  status: z
    .enum(["saved", "applied", "interviewing", "rejected", "offer", "not_interested"])
    .optional(),
  notes: z.string().max(2000).optional(),
});
