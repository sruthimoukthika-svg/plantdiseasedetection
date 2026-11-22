import Link from "next/link";
import { Disclaimer } from "@/components/Disclaimer";
import { getDiseases, KIND_LABEL, type PlantEntry } from "@/lib/api";

export const dynamic = "force-dynamic";

const KIND_ORDER = ["disease", "pest", "deficiency", "healthy"];

export default async function PlantsPage() {
  let crops: PlantEntry[] = [];
  let note = "";
  let error: string | null = null;
  try {
    const d = await getDiseases();
    crops = d.crops;
    note = d.confirmation_note;
  } catch (e) {
    error = e instanceof Error ? e.message : "unknown error";
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight">Supported Plants</h1>
      <p className="mt-2.5 max-w-2xl text-[15px] leading-relaxed text-ink-muted">
        This list is generated from the trained model&rsquo;s own class mapping, so it
        shows exactly what the model can predict — nothing is hard-coded here.
      </p>

      {error && (
        <div role="alert" className="mt-6 rounded-2xl border border-warn-200 bg-warn-50 p-5">
          <p className="font-semibold text-warn-700">Could not load the class list</p>
          <p className="mt-1.5 text-[14px] leading-relaxed text-warn-700">
            The analysis service is not reachable ({error}). Start the backend and
            reload this page.
          </p>
        </div>
      )}

      <div className="mt-8 space-y-8">
        {crops.map((c) => {
          const byKind = KIND_ORDER.map((k) => ({
            kind: k,
            items: c.conditions.filter((x) => x.kind === k),
          })).filter((g) => g.items.length > 0);
          return (
            <section key={c.crop} aria-labelledby={`crop-${c.crop}`}>
              <div className="flex items-baseline justify-between">
                <h2 id={`crop-${c.crop}`} className="text-2xl font-semibold tracking-tight">
                  {c.display}
                </h2>
                <span className="text-[13px] tabular-nums text-ink-faint">
                  {c.conditions.length} classes
                </span>
              </div>

              <div className="mt-4 space-y-5">
                {byKind.map((g) => (
                  <div key={g.kind}>
                    <h3 className="text-[12px] font-medium uppercase tracking-wider text-leaf-600">
                      {KIND_LABEL[g.kind] ?? g.kind}
                    </h3>
                    <div className="mt-2.5 grid gap-3 sm:grid-cols-2">
                      {g.items.map((x) => (
                        <div
                          key={x.condition}
                          className="rounded-xl border border-leaf-100 bg-surface p-4 shadow-card"
                        >
                          <div className="flex items-baseline justify-between gap-3">
                            <h4 className="font-medium tracking-tight">{x.display}</h4>
                            {x.pathogen && (
                              <span className="shrink-0 text-[11px] italic text-ink-faint">
                                {x.pathogen}
                              </span>
                            )}
                          </div>
                          {x.info?.summary && (
                            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-muted">
                              {x.info.summary}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      {crops.length > 0 && (
        <Disclaimer className="mt-10">
          <strong className="font-semibold">Not all of these are diseases.</strong>{" "}
          Spider mites and leaf miner are pest damage, and the three deficiency
          classes are nutrient disorders rather than infections — so the scanner
          reports a &ldquo;condition&rdquo;, not a disease. {note}
        </Disclaimer>
      )}

      <p className="mt-6 text-[13px] text-ink-faint">
        Measured accuracy per class is published on the{" "}
        <Link href="/about" className="text-leaf-600 underline underline-offset-2">
          About the AI
        </Link>{" "}
        page.
      </p>
    </div>
  );
}
