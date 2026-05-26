import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { Providers } from "./providers";
import { AppNav } from "@/components/app-nav";

export const metadata: Metadata = { title: "Finance Dashboard", description: "Investing research" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 antialiased">
        <Providers>
          <AppNav />
          {children}
        </Providers>
        <Analytics />
      </body>
    </html>
  );
}
