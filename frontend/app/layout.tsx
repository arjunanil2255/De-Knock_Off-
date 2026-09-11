import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SyncVerity — Deepfake Detection",
  description:
    "Cross-modal consistency-based deepfake detection via vector similarity scoring.",
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