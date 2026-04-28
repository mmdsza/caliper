import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Caliper — Grader Editor",
  description: "Author and iterate on grader / reward functions for RFT.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
