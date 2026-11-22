"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Disclaimer, ConfidenceBar } from "@/components/Disclaimer";
import { KIND_LABEL, scanImage, type ScanResult } from "@/lib/api";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_DIMENSION = 6000;

/** Phases reflect real work, not timed animation. */
type Phase =
  | { k: "idle" }
  | { k: "reading" }
  | { k: "uploading" }
  | { k: "analyzing" }
  | { k: "done"; result: ScanResult }
  | { k: "error"; message: string };

const PHASE_TEXT: Record<string, string> = {
  reading: "Reading image…",
  uploading: "Uploading image…",
  analyzing: "Validating and analyzing leaf…",
};

export default function ScannerPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [phase, setPhase] = useState<Phase>({ k: "idle" });
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = useCallback(async (f: File) => {
    setFileError(null);
    setPhase({ k: "idle" });
    if (!ACCEPTED.includes(f.type)) {
      setFileError("Please choose a JPG, PNG or WEBP image.");
      return;
    }
    if (f.size > MAX_BYTES) {
      setFileError(
        `That file is ${(f.size / 1024 / 1024).toFixed(1)} MB. The limit is 10 MB.`
      );
      return;
    }
    // reject absurd dimensions client-side as well as server-side
    const dims = await new Promise<{ w: number; h: number } | null>((res) => {
      const img = new Image();
      const u = URL.createObjectURL(f);
      img.onload = () => {
        res({ w: img.naturalWidth, h: img.naturalHeight });
        URL.revokeObjectURL(u);
      };
      img.onerror = () => {
        res(null);
        URL.revokeObjectURL(u);
      };
      img.src = u;
    });
    if (!dims) {
      setFileError("That file could not be read as an image.");
      return;
    }
    if (dims.w > MAX_DIMENSION || dims.h > MAX_DIMENSION) {
      setFileError(
        `That image is ${dims.w}×${dims.h}px. Please use one under ${MAX_DIMENSION}px on each side.`
      );
      return;
    }
    if (dims.w < 64 || dims.h < 64) {
      setFileError(
        `That image is only ${dims.w}×${dims.h}px — too small to analyse.`
      );
      return;
    }
    setFile(f);
  }, []);

  const reset = () => {
    setFile(null);
    setConfirmed(false);
    setPhase({ k: "idle" });
    setFileError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const run = async () => {
    if (!file) return;
    try {
      setPhase({ k: "reading" });
      setPhase({ k: "uploading" });
      const p = scanImage(file);
      setPhase({ k: "analyzing" });
      const result = await p;
      setPhase({ k: "done", result });
    } catch (e) {
      setPhase({
        k: "error",
        message:
          e instanceof Error
            ? `Could not reach the analysis service (${e.message}).`
            : "Could not reach the analysis service.",
      });
    }
  };

  const busy =
    phase.k === "reading" || phase.k === "uploading" || phase.k === "analyzing";
  const canScan = !!file && confirmed && !busy;

  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight">Disease Scanner</h1>
      <p className="mt-2.5 text-[15px] leading-relaxed text-ink-muted">
        Upload a clear photo of a single leaf. The scanner checks the image, identifies
        the plant, and predicts its condition.
      </p>

      <Disclaimer className="mt-6">
        <strong className="font-semibold">IMPORTANT:</strong> This scanner only
        supports tomato, potato and grape leaf images represented in its training
        datasets. Do not upload photographs of people, animals, unrelated objects
        or unsupported plants. Results may be inaccurate and should not be treated
        as a confirmed agricultural diagnosis.
      </Disclaimer>

      {phase.k !== "done" && (
        <section className="mt-6" aria-label="Upload image">
          {!file ? (
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                const f = e.dataTransfer.files?.[0];
                if (f) void accept(f);
              }}
              className={`rounded-2xl border-2 border-dashed p-10 text-center transition-colors sm:p-14 ${
                dragging
                  ? "border-leaf-400 bg-leaf-50"
                  : "border-leaf-200 bg-surface hover:border-leaf-300 hover:bg-leaf-50/40"
              }`}
            >
              <span
                aria-hidden
                className="mx-auto mb-4 grid h-14 w-14 place-items-center rounded-full bg-leaf-100 text-leaf-600"
              >
                <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 16V7m0 0L8.5 10.5M12 7l3.5 3.5" strokeLinecap="round" />
                  <path d="M4 15v2a3 3 0 003 3h10a3 3 0 003-3v-2" strokeLinecap="round" />
                </svg>
              </span>
              <p className="text-[15px] font-medium">
                Drag and drop a leaf image here
              </p>
              <p className="mt-1 text-[13px] text-ink-faint">
                JPG, PNG or WEBP · up to 10 MB
              </p>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="mt-5 rounded-xl border border-leaf-200 bg-surface px-4 py-2.5 text-sm font-medium text-leaf-700 transition-colors hover:bg-leaf-50"
              >
                Browse files
              </button>
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPTED.join(",")}
                className="sr-only"
                aria-label="Choose a leaf image"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void accept(f);
                }}
              />
            </div>
          ) : (
            <div className="rounded-2xl border border-leaf-100 bg-surface p-4 shadow-card sm:p-5">
              <div className="flex flex-col gap-4 sm:flex-row">
                {preview && (
                  <img
                    src={preview}
                    alt="Leaf you selected, awaiting analysis"
                    className="h-44 w-full rounded-xl object-cover sm:w-44"
                  />
                )}
                <div className="flex min-w-0 flex-1 flex-col justify-between">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{file.name}</p>
                    <p className="mt-1 text-[13px] text-ink-faint">
                      {(file.size / 1024).toFixed(0)} KB · {file.type.split("/")[1].toUpperCase()}
                    </p>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => inputRef.current?.click()}
                      disabled={busy}
                      className="rounded-lg border border-leaf-200 px-3 py-2 text-[13px] font-medium text-leaf-700 transition-colors hover:bg-leaf-50 disabled:opacity-50"
                    >
                      Replace image
                    </button>
                    <button
                      type="button"
                      onClick={reset}
                      disabled={busy}
                      className="rounded-lg px-3 py-2 text-[13px] font-medium text-ink-muted transition-colors hover:bg-leaf-50 disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                  <input
                    ref={inputRef}
                    type="file"
                    accept={ACCEPTED.join(",")}
                    className="sr-only"
                    aria-label="Replace leaf image"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void accept(f);
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {fileError && (
            <p role="alert" className="mt-3 text-[13px] font-medium text-warn-700">
              {fileError}
            </p>
          )}

          <label className="mt-5 flex cursor-pointer items-start gap-3 rounded-xl border border-leaf-100 bg-surface p-4">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
              disabled={busy}
              className="mt-0.5 h-4 w-4 shrink-0 rounded border-leaf-300 text-leaf-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-leaf-600"
            />
            <span className="text-[14px] leading-relaxed text-ink">
              I confirm that I am uploading a leaf image of a supported plant.
            </span>
          </label>

          <div className="mt-5 flex items-center gap-4">
            <button
              type="button"
              onClick={run}
              disabled={!canScan}
              className="rounded-xl bg-leaf-600 px-5 py-3 text-[15px] font-medium text-surface shadow-card transition-all hover:bg-leaf-700 hover:shadow-lift disabled:cursor-not-allowed disabled:bg-leaf-200 disabled:text-leaf-500 disabled:shadow-none"
            >
              {busy ? "Analyzing…" : "Scan leaf"}
            </button>
            {!canScan && !busy && (
              <p className="text-[13px] text-ink-faint">
                {!file
                  ? "Upload an image to continue."
                  : "Tick the confirmation to continue."}
              </p>
            )}
          </div>

          {busy && (
            <div
              role="status"
              aria-live="polite"
              className="mt-6 overflow-hidden rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card"
            >
              <p className="text-sm font-medium">{PHASE_TEXT[phase.k]}</p>
              <div className="relative mt-3 h-1.5 overflow-hidden rounded-full bg-leaf-100">
                <div className="absolute inset-y-0 w-1/3 animate-shimmer rounded-full bg-leaf-400" />
              </div>
              <p className="mt-2.5 text-[12px] text-ink-faint">
                Running on CPU — typically under a second.
              </p>
            </div>
          )}

          {phase.k === "error" && (
            <div
              role="alert"
              className="mt-6 rounded-2xl border border-warn-200 bg-warn-50 p-5 text-[14px] text-warn-700"
            >
              <p className="font-semibold">Analysis failed</p>
              <p className="mt-1.5 leading-relaxed">{phase.message}</p>
              <button
                type="button"
                onClick={run}
                className="mt-3 rounded-lg border border-warn-200 bg-surface px-3 py-2 text-[13px] font-medium"
              >
                Try again
              </button>
            </div>
          )}
        </section>
      )}

      {phase.k === "done" && (
        <Results
          result={phase.result}
          preview={preview}
          onReset={reset}
          onZoom={() => setZoom(true)}
        />
      )}

      {zoom && preview && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Enlarged image"
          onClick={() => setZoom(false)}
          className="fixed inset-0 z-50 grid place-items-center bg-ink/80 p-4"
        >
          <img
            src={preview}
            alt="Enlarged view of the leaf you uploaded"
            className="max-h-[88vh] max-w-full rounded-xl"
          />
          <button
            type="button"
            onClick={() => setZoom(false)}
            className="mt-4 rounded-lg bg-surface px-4 py-2 text-sm font-medium"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

function Results({
  result,
  preview,
  onReset,
  onZoom,
}: {
  result: ScanResult;
  preview: string | null;
  onReset: () => void;
  onZoom: () => void;
}) {
  // Rejected: show no prediction at all.
  if (!result.success) {
    return (
      <section className="mt-6 animate-fade-up" aria-label="Result">
        <div className="rounded-2xl border border-warn-200 bg-warn-50 p-6">
          <h2 className="text-xl font-semibold text-warn-700">Unsupported Image</h2>
          <p className="mt-2 text-[14px] leading-relaxed text-warn-700">
            {result.message}
          </p>
        </div>
        {preview && (
          <img
            src={preview}
            alt="The image you uploaded, which was not analysed"
            className="mt-4 h-48 w-48 rounded-xl object-cover"
          />
        )}
        <button
          type="button"
          onClick={onReset}
          className="mt-6 rounded-xl bg-leaf-600 px-5 py-3 text-[15px] font-medium text-surface transition-colors hover:bg-leaf-700"
        >
          Scan another image
        </button>
      </section>
    );
  }

  const r = result;
  return (
    <section className="mt-6 animate-fade-up" aria-label="Result">
      <div className="grid gap-5 sm:grid-cols-[200px_1fr]">
        <div>
          {preview && (
            <button
              type="button"
              onClick={onZoom}
              className="group relative block w-full overflow-hidden rounded-2xl"
              aria-label="Enlarge uploaded image"
            >
              <img
                src={preview}
                alt="The leaf you uploaded"
                className="h-48 w-full object-cover transition-transform duration-300 group-hover:scale-[1.03] sm:h-[200px]"
              />
              <span className="absolute bottom-2 right-2 rounded-md bg-ink/70 px-2 py-1 text-[11px] font-medium text-surface">
                Enlarge
              </span>
            </button>
          )}
        </div>

        <div className="rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card">
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full bg-leaf-100 px-2.5 py-1 text-[12px] font-medium text-leaf-800">
              {r.plant.name}
            </span>
            <span className="rounded-full border border-leaf-200 px-2.5 py-1 text-[12px] text-ink-muted">
              {KIND_LABEL[r.prediction.kind] ?? r.prediction.kind}
            </span>
            <span className="ml-auto text-[12px] tabular-nums text-ink-faint">
              {r.inference_ms} ms
            </span>
          </div>
          <h2 className="mt-3 text-2xl font-semibold tracking-tight">
            {r.prediction.name}
          </h2>
          <div className="mt-4 space-y-3">
            <ConfidenceBar value={r.prediction.confidence} label="Condition confidence" />
            <ConfidenceBar value={r.plant.confidence} label="Plant identification" />
          </div>
          <p className="mt-4 rounded-lg bg-leaf-50 p-3 text-[12px] leading-relaxed text-ink-muted">
            {r.confidence_note}
          </p>
        </div>
      </div>

      {r.alternatives.length > 0 && (
        <div className="mt-5 rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card">
          <h3 className="text-sm font-medium uppercase tracking-wider text-ink-faint">
            Other possibilities
          </h3>
          <ul className="mt-3 space-y-3">
            {r.alternatives.map((a) => (
              <li key={a.id}>
                <ConfidenceBar value={a.confidence} label={a.name} />
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[12px] text-ink-faint">
            Alternatives are always conditions of the identified plant — never of
            another species.
          </p>
        </div>
      )}

      <div className="mt-5 rounded-2xl border border-leaf-100 bg-surface p-5 shadow-card">
        <h3 className="text-lg font-semibold tracking-tight">
          About {r.prediction.name}
        </h3>
        {r.info.summary && (
          <p className="mt-2 text-[14px] leading-relaxed text-ink-muted">
            {r.info.summary}
          </p>
        )}
        {r.info.symptoms?.length > 0 && (
          <>
            <h4 className="mt-4 text-[13px] font-medium uppercase tracking-wider text-ink-faint">
              Common visible symptoms
            </h4>
            <ul className="mt-2 space-y-1.5">
              {r.info.symptoms.map((s) => (
                <li key={s} className="flex gap-2.5 text-[14px] leading-relaxed text-ink-muted">
                  <span aria-hidden className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-leaf-400" />
                  {s}
                </li>
              ))}
            </ul>
          </>
        )}
        {r.info.spreads && (
          <>
            <h4 className="mt-4 text-[13px] font-medium uppercase tracking-wider text-ink-faint">
              How it spreads
            </h4>
            <p className="mt-1.5 text-[14px] leading-relaxed text-ink-muted">
              {r.info.spreads}
            </p>
          </>
        )}
      </div>

      <Disclaimer className="mt-5">
        {r.info.confirmation_note} {r.disclaimer}
      </Disclaimer>

      <button
        type="button"
        onClick={onReset}
        className="mt-6 rounded-xl bg-leaf-600 px-5 py-3 text-[15px] font-medium text-surface shadow-card transition-all hover:bg-leaf-700 hover:shadow-lift"
      >
        Scan another image
      </button>
    </section>
  );
}
