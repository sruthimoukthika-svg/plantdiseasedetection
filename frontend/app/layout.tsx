import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "PlantHealth AI — Leaf Disease Classification",
  description:
    "Upload a leaf image and an AI model trained on public plant disease datasets will identify the plant and predict its condition. An experimental research tool, not a diagnosis service.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-dvh antialiased">
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <Nav />
        <main id="main">{children}</main>
        <footer className="mt-20 border-t border-leaf-100 bg-surface">
          <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
            <p className="text-[13px] leading-relaxed text-ink-muted">
              <strong className="font-medium text-ink">PlantHealth AI</strong> is an
              experimental AI classification tool for research and education. It is
              not a substitute for expert agricultural assessment, and its
              predictions are not confirmed diagnoses.
            </p>
            <p className="mt-3 text-[13px] text-ink-faint">
              Built on the PlantVillage, Tomato-Village, Potato Disease Leaf (PLD),
              AgML and PlantDoc datasets. Model accuracy is measured and published
              on the{" "}
              <a href="/about" className="text-leaf-600 underline underline-offset-2">
                About the AI
              </a>{" "}
              page, including its known weaknesses.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
