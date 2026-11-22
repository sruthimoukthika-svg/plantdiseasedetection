import Link from "next/link";
import { Disclaimer } from "@/components/Disclaimer";

const CROPS = [
  {
    name: "Tomato",
    conditions: 15,
    note: "Diseases, pest damage and nutrient deficiencies",
  },
  { name: "Potato", conditions: 3, note: "Early blight, late blight, healthy" },
  { name: "Grape", conditions: 4, note: "Black rot, Esca, leaf blight, healthy" },
];

const STEPS = [
  {
    n: "01",
    title: "Checks the image first",
    body: "Before any prediction, the image is checked for sharpness and for plant foliage, then scored against the patterns the model was trained on. Unrelated photos are turned away rather than labelled.",
  },
  {
    n: "02",
    title: "Identifies the plant",
    body: "A species classifier decides whether the leaf is tomato, potato or grape — or none of them. It has an explicit option to say the plant is unsupported.",
  },
  {
    n: "03",
    title: "Predicts the condition",
    body: "A classifier specific to the identified crop predicts the condition, so grape diseases can never be suggested for a potato leaf.",
  },
];

export default function Home() {
  return (
    <>
      <section className="mx-auto max-w-6xl px-4 pb-4 pt-14 sm:px-6 sm:pt-20">
        <div className="grid items-center gap-12 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="animate-fade-up">
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-leaf-200 bg-leaf-50 px-3 py-1 text-[12px] font-medium text-leaf-700">
              <span className="h-1.5 w-1.5 rounded-full bg-leaf-500" />
              AI research tool · Tomato, Potato, Grape
            </p>
            <h1 className="text-balance text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
              Understand Your Plant&rsquo;s Health with AI
            </h1>
            <p className="mt-5 max-w-xl text-[17px] leading-relaxed text-ink-muted">
              Upload a leaf image and let our AI analyze it for potential diseases
              using trained plant classification models.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/scanner"
                className="rounded-xl bg-leaf-600 px-5 py-3 text-[15px] font-medium text-surface shadow-card transition-all hover:bg-leaf-700 hover:shadow-lift"
              >
                Scan a Plant
              </Link>
              <Link
                href="/about"
                className="rounded-xl border border-leaf-200 px-5 py-3 text-[15px] font-medium text-leaf-700 transition-colors hover:bg-leaf-50"
              >
                How it works
              </Link>
            </div>
          </div>

          {/* Visual preview of the upload interface */}
          <div
            aria-hidden
            className="animate-fade-up rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card"
          >
            <div className="rounded-xl border-2 border-dashed border-leaf-200 bg-leaf-50/50 px-6 py-10 text-center">
              <span className="mx-auto mb-3 grid h-11 w-11 place-items-center rounded-full bg-leaf-100 text-leaf-600">
                <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 16V7m0 0L8.5 10.5M12 7l3.5 3.5" strokeLinecap="round" />
                  <path d="M4 15v2a3 3 0 003 3h10a3 3 0 003-3v-2" strokeLinecap="round" />
                </svg>
              </span>
              <p className="text-sm font-medium">Drag a leaf photo here</p>
              <p className="mt-1 text-[13px] text-ink-faint">JPG, PNG or WEBP</p>
            </div>
            <div className="mt-4 space-y-2.5">
              <div className="flex items-center justify-between rounded-lg bg-leaf-50 px-3.5 py-2.5">
                <span className="text-[13px] text-ink-muted">Identified plant</span>
                <span className="text-[13px] font-medium">Tomato</span>
              </div>
              <div className="rounded-lg bg-leaf-50 px-3.5 py-2.5">
                <div className="mb-1.5 flex items-center justify-between">
                  <span className="text-[13px] text-ink-muted">Predicted condition</span>
                  <span className="text-[13px] font-medium">Early Blight</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-leaf-100">
                  <div className="h-full w-[74%] rounded-full bg-leaf-500" />
                </div>
              </div>
            </div>
            <p className="mt-3 text-center text-[11px] text-ink-faint">
              Illustration of the scanner interface
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
        <h2 className="text-sm font-medium uppercase tracking-wider text-ink-faint">
          Supported plants
        </h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-3">
          {CROPS.map((c) => (
            <div
              key={c.name}
              className="rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card transition-shadow hover:shadow-lift"
            >
              <div className="flex items-baseline justify-between">
                <h3 className="text-lg font-semibold tracking-tight">{c.name}</h3>
                <span className="text-[13px] tabular-nums text-leaf-600">
                  {c.conditions} classes
                </span>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{c.note}</p>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[13px] text-ink-faint">
          Only these three crops are supported. The{" "}
          <Link href="/plants" className="text-leaf-600 underline underline-offset-2">
            Supported Plants
          </Link>{" "}
          page lists every condition the trained model can actually predict.
        </p>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
        <h2 className="text-2xl font-semibold tracking-tight">How the AI works</h2>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {STEPS.map((s) => (
            <div key={s.n} className="rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card">
              <span className="text-[12px] font-semibold tabular-nums text-leaf-400">
                {s.n}
              </span>
              <h3 className="mt-2 font-semibold tracking-tight">{s.title}</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-muted">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
        <h2 className="text-2xl font-semibold tracking-tight">Training data</h2>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-ink-muted">
          The model is trained on public research datasets. Each was inspected
          before use, and two were deliberately excluded from training.
        </p>
        <div className="mt-6 overflow-hidden rounded-2xl border border-leaf-100 bg-surface shadow-card">
          <table className="w-full text-left text-[13px]">
            <thead className="bg-leaf-50 text-ink-muted">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Dataset</th>
                <th scope="col" className="px-4 py-3 font-medium">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-leaf-100">
              <tr><td className="px-4 py-3 font-medium">PlantVillage</td><td className="px-4 py-3 text-ink-muted">Primary training data, all three crops</td></tr>
              <tr><td className="px-4 py-3 font-medium">Tomato-Village</td><td className="px-4 py-3 text-ink-muted">Real-world tomato images; adds pest and deficiency classes</td></tr>
              <tr><td className="px-4 py-3 font-medium">Potato Disease Leaf (PLD)</td><td className="px-4 py-3 text-ink-muted">Independent potato data; supplies healthy-leaf images</td></tr>
              <tr><td className="px-4 py-3 font-medium">AgML Potato</td><td className="px-4 py-3 text-ink-muted">Evaluation only — labels are pathogen groups, not named diseases</td></tr>
              <tr><td className="px-4 py-3 font-medium">PlantDoc</td><td className="px-4 py-3 text-ink-muted">Evaluation only — real-world generalisation</td></tr>
              <tr><td className="px-4 py-3 font-medium">Grape (augmented)</td><td className="px-4 py-3 text-ink-muted">Excluded — duplicates PlantVillage grape, with split leakage</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 pb-6 sm:px-6">
        <Disclaimer>
          <strong className="font-semibold">Limitations.</strong> This is an AI
          research and classification tool, not a professional agricultural
          diagnosis service. It supports only tomato, potato and grape leaves
          represented in its training datasets. Accuracy is substantially lower on
          real-world field photographs than on the standardised laboratory images
          that make up most of its training data — the{" "}
          <Link href="/about" className="underline underline-offset-2">
            About the AI
          </Link>{" "}
          page publishes both figures. Predictions require confirmation by someone
          qualified.
        </Disclaimer>
      </section>
    </>
  );
}
