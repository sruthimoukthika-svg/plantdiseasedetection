import Link from "next/link";
import { Disclaimer } from "@/components/Disclaimer";
import { getModelInfo } from "@/lib/api";

export const dynamic = "force-dynamic";

const DATASETS = [
  { name: "PlantVillage", href: "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset", role: "Primary training data for all three crops" },
  { name: "Tomato-Village", href: "https://github.com/mamta-joshi-gehlot/Tomato-Village", role: "Real-world tomato images; source of the pest and nutrient-deficiency classes" },
  { name: "Potato Disease Leaf (PLD)", href: "https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld", role: "Independent potato data; supplies the healthy-potato images PlantVillage lacks" },
  { name: "AgML Potato", href: "https://huggingface.co/datasets/Project-AgML/potato_leaf_disease_classification", role: "Evaluation only — its labels are pathogen groups, not named diseases" },
  { name: "PlantDoc", href: "https://github.com/pratikkayal/PlantDoc-Dataset", role: "Evaluation only — real-world generalisation and rejection testing" },
  { name: "Grape leaf disease (augmented)", href: "https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset", role: "Excluded — augmented copies of PlantVillage grape with train/test leakage" },
];

function pct(v: unknown) {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

export default async function AboutPage() {
  let m: Record<string, any> | null = null;
  let error: string | null = null;
  try {
    m = await getModelInfo();
  } catch (e) {
    error = e instanceof Error ? e.message : "unknown error";
  }
  const meas = m?.measured ?? {};
  const src = meas.per_source_dataset ?? {};
  const rej = m?.rejection ?? {};
  const far = rej.false_accept_rate_by_set ?? {};

  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight">About the AI</h1>
      <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
        What this system does, how it was built and measured, and where it fails.
        Every figure on this page is read live from the model&rsquo;s own evaluation
        artefacts — none of it is written by hand.
      </p>

      {error && (
        <div role="alert" className="mt-6 rounded-2xl border border-warn-200 bg-warn-50 p-5 text-[14px] text-warn-700">
          <p className="font-semibold">Live metrics unavailable</p>
          <p className="mt-1.5">The analysis service is not reachable ({error}).</p>
        </div>
      )}

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">What it does</h2>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
          You upload a photograph of a leaf. The system checks the image is usable,
          decides whether the leaf belongs to one of three supported crops — tomato,
          potato or grape — and if so predicts its condition and shows a confidence
          score. If the image is not a supported leaf, it says so and makes no
          prediction at all.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">How classification works</h2>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
          An ImageNet-pretrained <strong className="font-medium text-ink">{m?.backbone ?? "EfficientNet-B0"}</strong>{" "}
          converts the image into a feature vector. That backbone is frozen —
          its weights are not altered during training. Two kinds of classifier sit
          on top:
        </p>
        <ol className="mt-4 space-y-3">
          {[
            ["A species gate", "decides tomato, potato, grape — or unsupported. The fourth option matters: a three-way classifier would be forced to call an apple leaf one of the three crops."],
            ["One condition head per crop", "predicts the condition using only that crop's classes. This makes it structurally impossible to suggest a grape disease for a potato leaf, and it is why the alternatives you see never come from another species."],
          ].map(([t, b]) => (
            <li key={t} className="rounded-xl border border-leaf-100 bg-surface p-4">
              <p className="text-[14px] leading-relaxed text-ink-muted">
                <strong className="font-medium text-ink">{t}</strong> {b}
              </p>
            </li>
          ))}
        </ol>
        <p className="mt-4 text-[15px] leading-relaxed text-ink-muted">
          Two models were trained and compared; the one published here was selected
          on validation macro F1, with the test set untouched during selection.
          Macro F1 was used rather than accuracy because the classes are very
          unevenly sized, and accuracy would have looked healthy while the smallest
          classes failed.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">Measured performance</h2>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
          Measured on {meas ? (m?.test_set_size ?? "—") : "—"} held-out test images
          that were never used for training or model selection.
        </p>
        <dl className="mt-5 grid gap-3 sm:grid-cols-2">
          {[
            ["End-to-end accuracy", pct(meas.end_to_end_accuracy)],
            ["End-to-end macro F1", pct(meas.end_to_end_macro_f1)],
            ["Species identification", pct(meas.species_gate_accuracy)],
            ["Inference time", meas.inference_ms_per_image?.mean ? `${meas.inference_ms_per_image.mean} ms` : "—"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-xl border border-leaf-100 bg-surface p-4">
              <dt className="text-[13px] text-ink-muted">{k}</dt>
              <dd className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{v}</dd>
            </div>
          ))}
        </dl>

        <div className="mt-6 rounded-2xl border border-warn-200 bg-warn-50 p-5">
          <h3 className="font-semibold text-warn-700">
            That headline number is misleading on its own
          </h3>
          <p className="mt-2 text-[14px] leading-relaxed text-warn-700">
            Most training images are standardised laboratory photographs — a single
            leaf, even lighting, plain background. Real photographs taken in a field
            are much harder, and the model is markedly worse on them. Accuracy split
            by where the images came from:
          </p>
          <dl className="mt-4 space-y-2">
            {[
              ["PlantVillage — laboratory images", src.plantvillage?.end_to_end_accuracy],
              ["Potato PLD", src.potato_pld?.end_to_end_accuracy],
              ["Tomato-Village — real field photographs", src.tomato_village?.end_to_end_accuracy],
            ].map(([k, v]) => (
              <div key={String(k)} className="flex items-baseline justify-between gap-4 border-b border-warn-200 pb-2 last:border-0">
                <dt className="text-[14px] text-warn-700">{k}</dt>
                <dd className="text-[15px] font-semibold tabular-nums text-warn-700">{pct(v)}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-[14px] leading-relaxed text-warn-700">
            If you photograph a leaf outdoors on a phone, the field figure is the one
            that predicts your experience — not the headline.
          </p>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          How unsupported images are rejected
        </h2>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
          A classifier trained on three crops will still return one of them for a
          photograph of a cat, with a high score, because a high score only means
          &ldquo;most likely of the options I know&rdquo;. A confident-looking number is
          therefore not evidence the image is supported. Three independent layers
          guard against this, and a high score alone is deliberately never used:
        </p>
        <ul className="mt-4 space-y-2.5">
          {[
            "Image checks before the model: sharpness, and whether enough of the picture looks like plant tissue at all.",
            "An explicit unsupported class in the species gate, trained on leaves of other species.",
            "A distance measure comparing the image against the patterns seen in training, plus an energy score — both calibrated on validation data.",
          ].map((t) => (
            <li key={t} className="flex gap-2.5 text-[14px] leading-relaxed text-ink-muted">
              <span aria-hidden className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-leaf-400" />
              {t}
            </li>
          ))}
        </ul>
        <p className="mt-4 text-[14px] leading-relaxed text-ink-muted">
          Measured share of unsupported images wrongly accepted, by category — lower
          is better:
        </p>
        <dl className="mt-3 space-y-2">
          {[
            ["Random non-plant photographs", far.non_plant_objects],
            ["Unsupported species, field photographs", far.unsupported_field],
            ["Unsupported leaf species never seen in training", far.unsupported_lab_heldout],
          ].map(([k, v]) => (
            <div key={String(k)} className="flex items-baseline justify-between gap-4 rounded-xl border border-leaf-100 bg-surface px-4 py-3">
              <dt className="text-[14px] text-ink-muted">{k}</dt>
              <dd className="text-[15px] font-semibold tabular-nums">{pct(v)}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-[13px] leading-relaxed text-ink-faint">
          The thresholds are set to accept about 90% of genuine supported leaves.
          Pushing that to 95% would roughly double the number of unsupported images
          wrongly accepted, so the system is deliberately tuned to turn away a
          borderline real leaf rather than guess at an unsupported one.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">What confidence means</h2>
        <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
          Confidence is the model&rsquo;s own score for that class on that image. It is{" "}
          <strong className="font-medium text-ink">not</strong> the probability that
          the plant truly has the condition. A 95% score does not mean 95% certainty
          of disease; it means the model ranked that class far above the others.
          Scores are also uncalibrated, so they tend to read higher than the model&rsquo;s
          real reliability.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">Known limitations</h2>
        <ul className="mt-3 space-y-2.5">
          {[
            "Much weaker on real-world field photographs than on laboratory images — see the split above.",
            "Rejecting leaf species it was never trained to reject is the weakest part of the system. Random objects are turned away reliably; an unfamiliar plant species is not.",
            "Only tomato, potato and grape are supported. Every other plant should be rejected, and sometimes will not be.",
            "Some classes rest on very little data. Any class flagged with low support has an accuracy figure too uncertain to rely on.",
            "The model sees a single leaf in isolation. It cannot weigh the whole plant, growing conditions, climate or history — all of which a real diagnosis depends on.",
            "Several conditions look alike early on, and non-disease causes such as watering, heat or spray damage can mimic disease symptoms.",
            "Confidence scores are uncalibrated and read optimistically.",
          ].map((t) => (
            <li key={t} className="flex gap-2.5 text-[14px] leading-relaxed text-ink-muted">
              <span aria-hidden className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-warn-200" />
              {t}
            </li>
          ))}
        </ul>
        {m?.low_support_classes && Object.keys(m.low_support_classes).length > 0 && (
          <p className="mt-4 rounded-xl bg-leaf-50 p-4 text-[13px] leading-relaxed text-ink-muted">
            <strong className="font-medium text-ink">Low-support classes:</strong>{" "}
            {Object.entries(m.low_support_classes)
              .map(([k, v]) => `${k} (${v} test images)`)
              .join(", ")}
            . Per-class figures for these are not statistically meaningful.
          </p>
        )}
      </section>

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">Datasets used</h2>
        <div className="mt-4 overflow-hidden rounded-2xl border border-leaf-100 bg-surface shadow-card">
          <table className="w-full text-left text-[13px]">
            <thead className="bg-leaf-50 text-ink-muted">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Dataset</th>
                <th scope="col" className="px-4 py-3 font-medium">Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-leaf-100">
              {DATASETS.map((d) => (
                <tr key={d.name}>
                  <td className="px-4 py-3 align-top">
                    <a href={d.href} target="_blank" rel="noopener noreferrer"
                       className="font-medium text-leaf-700 underline underline-offset-2">
                      {d.name}
                    </a>
                  </td>
                  <td className="px-4 py-3 align-top text-ink-muted">{d.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Disclaimer className="mt-10">
        <strong className="font-semibold">
          This is an experimental AI classification tool, not a substitute for
          expert agricultural assessment.
        </strong>{" "}
        It is intended for research and education. Do not make treatment, spraying
        or crop-destruction decisions based on its output. Consult a qualified
        agronomist or plant pathologist, who can also arrange laboratory testing
        where it matters.
      </Disclaimer>

      <p className="mt-6 text-[13px] text-ink-faint">
        See the{" "}
        <Link href="/plants" className="text-leaf-600 underline underline-offset-2">
          Supported Plants
        </Link>{" "}
        page for every condition the model can predict.
      </p>
    </div>
  );
}
