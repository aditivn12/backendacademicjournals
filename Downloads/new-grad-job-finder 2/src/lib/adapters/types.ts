import { RawJobListing, SearchParams } from "@/lib/types";

export interface JobSourceAdapter {
  sourceName: string;
  searchJobs(params: SearchParams): Promise<RawJobListing[]>;
  isAvailable(): Promise<boolean>;
}
