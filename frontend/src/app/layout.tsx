import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MODEL SHIFT — A QLoRA experiment",
  description: "A visual before/after laboratory: explore 70 recorded responses, measured changes, and the limitations of a Qwen2.5 QLoRA experiment.",
  openGraph: {
    title: "MODEL SHIFT — Base → Δ → Fine-tuned",
    description: "One model. 70 prompts. A measured shift in paraphrase recall. Explore the evidence, including the failures.",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
