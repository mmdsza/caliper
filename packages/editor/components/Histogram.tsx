"use client";

interface HistogramProps {
  scores: number[];
  bins?: number;
}

export function Histogram({ scores, bins = 10 }: HistogramProps) {
  if (scores.length === 0) {
    return (
      <div style={{ color: "#9a9a9a", fontSize: 12 }}>(no scored rows)</div>
    );
  }

  const counts = new Array<number>(bins).fill(0);
  for (const s of scores) {
    const clamped = Math.max(0, Math.min(0.9999999, s));
    const idx = Math.floor(clamped * bins);
    counts[idx] = (counts[idx] ?? 0) + 1;
  }
  const max = Math.max(...counts, 1);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "60px 1fr 30px",
        rowGap: 2,
        fontFamily:
          "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        fontSize: 11,
        color: "#cfcfcf",
      }}
    >
      {counts.map((c, i) => {
        const lo = (i / bins).toFixed(1);
        const hi = ((i + 1) / bins).toFixed(1);
        const widthPct = (c / max) * 100;
        return (
          <Row
            key={i}
            label={`${lo}–${hi}`}
            count={c}
            widthPct={widthPct}
          />
        );
      })}
    </div>
  );
}

function Row({
  label,
  count,
  widthPct,
}: {
  label: string;
  count: number;
  widthPct: number;
}) {
  return (
    <>
      <div style={{ color: "#9a9a9a", textAlign: "right", paddingRight: 6 }}>
        {label}
      </div>
      <div style={{ background: "#262626", height: 14 }}>
        <div
          style={{
            background: "#4a90d9",
            height: "100%",
            width: `${widthPct}%`,
            transition: "width 120ms",
          }}
        />
      </div>
      <div style={{ color: "#9a9a9a", paddingLeft: 6 }}>{count}</div>
    </>
  );
}

export default Histogram;
