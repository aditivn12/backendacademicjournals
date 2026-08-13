import { JobSourceAdapter } from "@/lib/adapters/types";
import { RawJobListing, SearchParams } from "@/lib/types";

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString();
}

// 30+ seeded roles per spec section 14, deliberately including:
// - The 3 default test companies (JPMorgan Chase, Salesforce, Citibank)
//   so you can verify end-to-end that real target companies surface.
// - One clearly senior role (should be excluded by default).
// - A duplicate listing from two different mock sources (dedup check).
// - A remote role tied to a target-region employer.
const MOCK_JOBS: RawJobListing[] = [
  // --- JPMorgan Chase (test company) ---
  {
    source: "mock",
    sourceUrl: "https://careers.jpmorgan.com/jobs/1001",
    applicationUrl: "https://careers.jpmorgan.com/jobs/1001",
    companyName: "JPMorgan Chase",
    title: "Technology Analyst Program - Software Engineering",
    description: "New grad program for software engineers joining our technology analyst track.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(3),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.jpmorgan.com/jobs/1002",
    applicationUrl: "https://careers.jpmorgan.com/jobs/1002",
    companyName: "JPMorgan Chase",
    title: "Risk Technology Analyst, Entry Level",
    description: "Entry level analyst supporting risk technology platforms.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(6),
  },
  // --- Salesforce (test company) ---
  {
    source: "mock",
    sourceUrl: "https://careers.salesforce.com/jobs/2001",
    applicationUrl: "https://careers.salesforce.com/jobs/2001",
    companyName: "Salesforce",
    title: "Associate Software Engineer, University Grad",
    description: "University graduate program for early career software engineers.",
    rawLocation: "San Francisco, CA",
    postedAt: daysAgo(1),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.salesforce.com/jobs/2002",
    applicationUrl: "https://careers.salesforce.com/jobs/2002",
    companyName: "Salesforce",
    title: "Data Analyst I",
    description: "Entry level data analyst role supporting product analytics.",
    rawLocation: "San Francisco, CA · Hybrid",
    postedAt: daysAgo(10),
  },
  // --- Citibank (test company) ---
  {
    source: "mock",
    sourceUrl: "https://jobs.citi.com/jobs/3001",
    applicationUrl: "https://jobs.citi.com/jobs/3001",
    companyName: "Citibank",
    title: "New Grad Risk Analyst",
    description: "Rotational risk analyst program for new graduates.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(2),
  },
  {
    source: "mock",
    sourceUrl: "https://jobs.citi.com/jobs/3002",
    applicationUrl: "https://jobs.citi.com/jobs/3002",
    companyName: "Citibank",
    title: "Software Engineer I",
    description: "Entry level software engineer on our consumer banking platform team.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(14),
  },
  // --- Dedup test pair: same Citibank role appearing via a second source ---
  {
    source: "mock-secondary",
    sourceUrl: "https://www.linkedin.com/jobs/view/3002-mirror",
    applicationUrl: "https://jobs.citi.com/jobs/3002",
    companyName: "Citibank",
    title: "Software Engineer I",
    description: "Entry level software engineer on our consumer banking platform team.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(14),
  },
  // --- Other financial/fintech ---
  {
    source: "mock",
    sourceUrl: "https://www.capitalonecareers.com/jobs/4001",
    applicationUrl: "https://www.capitalonecareers.com/jobs/4001",
    companyName: "Capital One",
    title: "Software Engineer I - Risk Platform",
    description: "New grad software engineer building risk decisioning platforms.",
    rawLocation: "New York, NY · Hybrid",
    postedAt: daysAgo(2),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.moodys.com/jobs/4002",
    applicationUrl: "https://careers.moodys.com/jobs/4002",
    companyName: "Moody's",
    title: "Data Analyst, New Grad",
    description: "Entry level data analyst supporting credit risk analytics.",
    rawLocation: "Raleigh, NC",
    postedAt: daysAgo(5),
  },
  {
    source: "mock",
    sourceUrl: "https://www.bloomberg.com/careers/jobs/4003",
    applicationUrl: "https://www.bloomberg.com/careers/jobs/4003",
    companyName: "Bloomberg",
    title: "Software Engineer, University Recruiting",
    description: "Entry level software engineer role for recent graduates.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(4),
  },
  {
    source: "mock",
    sourceUrl: "https://stripe.com/jobs/4004",
    applicationUrl: "https://stripe.com/jobs/4004",
    companyName: "Stripe",
    title: "New Grad Software Engineer",
    description: "Early career software engineer on payments infrastructure.",
    rawLocation: "San Francisco, CA",
    postedAt: daysAgo(8),
  },
  {
    source: "mock",
    sourceUrl: "https://www.affirm.com/careers/jobs/4005",
    applicationUrl: "https://www.affirm.com/careers/jobs/4005",
    companyName: "Affirm",
    title: "Quantitative Analyst, Entry Level - Credit Risk",
    description: "Entry level quantitative analyst supporting credit risk models.",
    rawLocation: "San Francisco, CA",
    postedAt: daysAgo(9),
  },
  // --- Big tech, non-fintech ---
  {
    source: "mock",
    sourceUrl: "https://careers.google.com/jobs/5001",
    applicationUrl: "https://careers.google.com/jobs/5001",
    companyName: "Google",
    title: "Software Engineer III", // note: should be filtered as too senior-sounding title-wise? left entry via desc
    description: "New grad software engineer, university graduate program, 0-2 years experience.",
    rawLocation: "Seattle, WA",
    postedAt: daysAgo(3),
  },
  {
    source: "mock",
    sourceUrl: "https://www.amazon.jobs/jobs/5002",
    applicationUrl: "https://www.amazon.jobs/jobs/5002",
    companyName: "Amazon",
    title: "Data Scientist I",
    description: "Entry level data scientist role, new grad hiring.",
    rawLocation: "Seattle, WA",
    postedAt: daysAgo(1),
  },
  {
    source: "mock",
    sourceUrl: "https://www.metacareers.com/jobs/5003",
    applicationUrl: "https://www.metacareers.com/jobs/5003",
    companyName: "Meta",
    title: "University Graduate - Machine Learning Engineer",
    description: "Entry level machine learning engineer for recent university graduates.",
    rawLocation: "Menlo Park, CA",
    postedAt: daysAgo(6),
  },
  {
    source: "mock",
    sourceUrl: "https://www.snowflake.com/careers/jobs/5004",
    applicationUrl: "https://www.snowflake.com/careers/jobs/5004",
    companyName: "Snowflake",
    title: "New Grad Data Engineer",
    description: "Early career data engineer joining our platform team.",
    rawLocation: "San Mateo, CA",
    postedAt: daysAgo(11),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.datadoghq.com/jobs/5005",
    applicationUrl: "https://careers.datadoghq.com/jobs/5005",
    companyName: "Datadog",
    title: "Associate Software Engineer",
    description: "Entry level software engineer, early career program.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(7),
  },
  // --- Chicago ---
  {
    source: "mock",
    sourceUrl: "https://www.aon.com/careers/jobs/6001",
    applicationUrl: "https://www.aon.com/careers/jobs/6001",
    companyName: "Aon",
    title: "Risk Analyst, Entry Level",
    description: "Entry level risk analyst joining our Chicago risk consulting practice.",
    rawLocation: "Chicago, IL",
    postedAt: daysAgo(4),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.travelers.com/jobs/6002",
    applicationUrl: "https://careers.travelers.com/jobs/6002",
    companyName: "Travelers",
    title: "Entry Level Data Scientist",
    description: "New graduate data scientist role in Chicago.",
    rawLocation: "Chicago, IL",
    postedAt: daysAgo(2),
  },
  {
    source: "mock",
    sourceUrl: "https://www.allstate.com/careers/jobs/6003",
    applicationUrl: "https://www.allstate.com/careers/jobs/6003",
    companyName: "Allstate",
    title: "Technology Rotational Program Associate",
    description: "General technology rotational program for new graduates.",
    rawLocation: "Chicago, IL",
    postedAt: daysAgo(13),
  },
  // --- Seattle ---
  {
    source: "mock",
    sourceUrl: "https://careers.microsoft.com/jobs/7001",
    applicationUrl: "https://careers.microsoft.com/jobs/7001",
    companyName: "Microsoft",
    title: "Software Engineer, University Graduate",
    description: "University graduate program for software engineers.",
    rawLocation: "Redmond, WA",
    postedAt: daysAgo(5),
  },
  {
    source: "mock",
    sourceUrl: "https://www.amazon.jobs/jobs/7002",
    applicationUrl: "https://www.amazon.jobs/jobs/7002",
    companyName: "Amazon",
    title: "Business Intelligence Analyst I",
    description: "Entry level business intelligence analyst role.",
    rawLocation: "Bellevue, WA",
    postedAt: daysAgo(9),
  },
  // --- North Carolina ---
  {
    source: "mock",
    sourceUrl: "https://www.sas.com/careers/jobs/8001",
    applicationUrl: "https://www.sas.com/careers/jobs/8001",
    companyName: "SAS",
    title: "New Grad Data Scientist",
    description: "New graduate data scientist role at our Cary headquarters.",
    rawLocation: "Cary, NC",
    postedAt: daysAgo(1),
  },
  {
    source: "mock",
    sourceUrl: "https://www.redhat.com/jobs/8002",
    applicationUrl: "https://www.redhat.com/jobs/8002",
    companyName: "Red Hat",
    title: "Associate Cloud Engineer",
    description: "Entry level cloud/infrastructure engineer, early career.",
    rawLocation: "Raleigh, NC",
    postedAt: daysAgo(6),
  },
  {
    source: "mock",
    sourceUrl: "https://jobs.iqvia.com/jobs/8003",
    applicationUrl: "https://jobs.iqvia.com/jobs/8003",
    companyName: "IQVIA",
    title: "Data Analyst, Entry Level",
    description: "Entry level data analyst role supporting healthcare data analytics.",
    rawLocation: "Durham, NC",
    postedAt: daysAgo(8),
  },
  {
    source: "mock",
    sourceUrl: "https://careers.bankofamerica.com/jobs/8004",
    applicationUrl: "https://careers.bankofamerica.com/jobs/8004",
    companyName: "Bank of America",
    title: "Cybersecurity Analyst, Entry Level",
    description: "Entry level cybersecurity analyst role, new grad hiring.",
    rawLocation: "Charlotte, NC",
    postedAt: daysAgo(3),
  },
  // --- Remote role tied to a target-region employer ---
  {
    source: "mock",
    sourceUrl: "https://www.databricks.com/careers/jobs/9001",
    applicationUrl: "https://www.databricks.com/careers/jobs/9001",
    companyName: "Databricks",
    title: "New Grad Machine Learning Engineer (Remote)",
    description:
      "Remote machine learning engineer role. Databricks has an office in San Francisco, CA.",
    rawLocation: "Remote - US",
    postedAt: daysAgo(2),
  },
  // --- Clearly senior role that should be auto-excluded ---
  {
    source: "mock",
    sourceUrl: "https://careers.google.com/jobs/9999",
    applicationUrl: "https://careers.google.com/jobs/9999",
    companyName: "Google",
    title: "Senior Staff Software Engineer, Infrastructure",
    description: "Senior staff-level role requiring 10+ years of experience leading engineering teams.",
    rawLocation: "Mountain View, CA",
    postedAt: daysAgo(1),
  },
  // --- Internship (should only appear if includeInternships is enabled) ---
  {
    source: "mock",
    sourceUrl: "https://www.wellsfargojobs.com/jobs/9998",
    applicationUrl: "https://www.wellsfargojobs.com/jobs/9998",
    companyName: "Wells Fargo",
    title: "Summer Risk Management Intern",
    description: "Summer internship in the risk management division.",
    rawLocation: "Charlotte, NC",
    postedAt: daysAgo(20),
  },
  // --- Contract-only (should only appear if includeContractRoles enabled) ---
  {
    source: "mock",
    sourceUrl: "https://www.accenture.com/careers/jobs/9997",
    applicationUrl: "https://www.accenture.com/careers/jobs/9997",
    companyName: "Accenture",
    title: "Technology Analyst - Contract",
    description: "Contract-only entry level technology analyst position, 6 month contract.",
    employmentType: "Contract",
    rawLocation: "Chicago, IL",
    postedAt: daysAgo(15),
  },
  {
    source: "mock",
    sourceUrl: "https://www.oracle.com/careers/jobs/9996",
    applicationUrl: "https://www.oracle.com/careers/jobs/9996",
    companyName: "Oracle",
    title: "Cloud Infrastructure Engineer, Entry Level",
    description: "Entry level cloud engineer role, new grad hiring welcome.",
    rawLocation: "San Francisco, CA",
    postedAt: daysAgo(12),
  },
  {
    source: "mock",
    sourceUrl: "https://www.mongodb.com/careers/jobs/9995",
    applicationUrl: "https://www.mongodb.com/careers/jobs/9995",
    companyName: "MongoDB",
    title: "Software Engineer I, New Grad",
    description: "New grad software engineer role on our core database team.",
    rawLocation: "New York, NY",
    postedAt: daysAgo(4),
  },
];

export class MockAdapter implements JobSourceAdapter {
  sourceName = "mock";

  async isAvailable(): Promise<boolean> {
    return true;
  }

  async searchJobs(_params: SearchParams): Promise<RawJobListing[]> {
    return MOCK_JOBS;
  }
}
