export function Disclaimer({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      role="note"
      className={`rounded-xl border border-warn-200 bg-warn-50 p-4 text-[13px] leading-relaxed text-warn-700 ${className}`}
    >
      {children}
    </div>
  );
}

export function ConfidenceBar({
  value,
  label,
}: {
  value: number;
  label?: string;
}) {
  const pct = Math.round(value * 1000) / 10;
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        {label && <span className="text-[13px] text-ink-muted">{label}</span>}
        <span className="text-[13px] font-medium tabular-nums text-ink">
          {pct.toFixed(1)}%
        </span>
      </div>
      <div
        className="h-2 overflow-hidden rounded-full bg-leaf-100"
        role="meter"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ?? "Confidence"}
      >
        <div
          className="h-full rounded-full bg-leaf-500 transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(pct, 1.5)}%` }}
        />
      </div>
    </div>
  );
}
