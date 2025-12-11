import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Polars NLQ Workspace",
  description: "Chat-style transformations with Polars and SQL",
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

