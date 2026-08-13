import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "New Grad Job Finder",
  description: "Discover entry-level CS, data, fintech, and risk-tech roles.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
