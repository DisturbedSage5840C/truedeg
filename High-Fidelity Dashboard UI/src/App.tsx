import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts";

// ─── Data ────────────────────────────────────────────────────────────────────

const degData = [
  { lap: 1, naive: 0.02, clean: 0.018 },
  { lap: 2, naive: 0.06, clean: 0.05 },
  { lap: 3, naive: 0.05, clean: 0.082 },
  { lap: 4, naive: 0.13, clean: 0.11 },
  { lap: 5, naive: 0.17, clean: 0.143 },
  { lap: 6, naive: 0.19, clean: 0.175 },
  { lap: 7, naive: 0.26, clean: 0.206 },
  { lap: 8, naive: 0.21, clean: 0.238 },
  { lap: 9, naive: 0.31, clean: 0.27 },
  { lap: 10, naive: 0.35, clean: 0.302 },
  { lap: 11, naive: 0.38, clean: 0.334 },
  { lap: 12, naive: 0.42, clean: 0.366 },
  { lap: 13, naive: 0.46, clean: 0.398 },
  { lap: 14, naive: 0.44, clean: 0.432 },
  { lap: 15, naive: 0.52, clean: 0.466 },
  { lap: 16, naive: 0.57, clean: 0.502 },
  { lap: 17, naive: 0.60, clean: 0.54 },
  { lap: 18, naive: 0.68, clean: 0.58 },
  { lap: 19, naive: 0.72, clean: 0.622 },
  { lap: 20, naive: 0.76, clean: 0.668 },
  { lap: 21, naive: 0.81, clean: 0.718 },
  { lap: 22, naive: 0.88, clean: 0.774 },
  { lap: 23, naive: 0.97, clean: 0.834 },
  { lap: 24, naive: 1.06, clean: 0.902 },
  { lap: 25, naive: 1.08, clean: 0.978 },
  { lap: 26, naive: 1.19, clean: 1.062 },
  { lap: 27, naive: 1.28, clean: 1.154 },
  { lap: 28, naive: 1.38, clean: 1.256 },
  { lap: 29, naive: 1.52, clean: 1.368 },
  { lap: 30, naive: 1.69, clean: 1.492 },
];

const scatterNormal = [
  { x: 18.2, y: 142 }, { x: 19.1, y: 138 }, { x: 17.8, y: 145 },
  { x: 20.3, y: 135 }, { x: 18.9, y: 141 }, { x: 19.6, y: 137 },
  { x: 17.5, y: 148 }, { x: 20.1, y: 133 }, { x: 18.4, y: 144 },
  { x: 19.8, y: 136 }, { x: 18.0, y: 146 }, { x: 21.0, y: 130 },
  { x: 17.2, y: 151 }, { x: 19.3, y: 139 }, { x: 20.7, y: 131 },
  { x: 18.7, y: 143 }, { x: 19.5, y: 138 }, { x: 17.9, y: 147 },
  { x: 20.4, y: 134 }, { x: 18.6, y: 142 }, { x: 19.0, y: 140 },
  { x: 21.2, y: 128 }, { x: 17.6, y: 149 }, { x: 18.3, y: 144 },
  { x: 20.0, y: 132 }, { x: 19.7, y: 136 }, { x: 18.5, y: 143 },
];

const scatterOutlier = [
  { x: 22.8, y: 118 },
  { x: 23.4, y: 112 },
  { x: 24.1, y: 108 },
];

// ─── Sub-components ───────────────────────────────────────────────────────────

function Label({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={`text-[9px] font-mono tracking-[0.18em] uppercase ${className}`}
      style={{ color: "var(--text-label)", fontFamily: "var(--mono)" }}
    >
      {children}
    </span>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <span
        className="text-[9px] tracking-[0.22em] font-semibold uppercase"
        style={{ color: "var(--text-muted)", fontFamily: "var(--mono)" }}
      >
        {children}
      </span>
      <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
    </div>
  );
}

function Divider() {
  return <div className="h-px my-4" style={{ background: "var(--border)" }} />;
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

function DegTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const naive = payload.find((p: any) => p.dataKey === "naive")?.value;
  const clean = payload.find((p: any) => p.dataKey === "clean")?.value;
  const delta = naive && clean ? (naive - clean).toFixed(3) : null;
  return (
    <div
      className="px-3 py-2"
      style={{
        background: "#1a1a1b",
        border: "1px solid rgba(255,255,255,0.14)",
        fontFamily: "var(--mono)",
      }}
    >
      <div className="text-[9px] tracking-widest mb-2" style={{ color: "var(--text-label)" }}>
        LAP {label}
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-4">
          <span className="text-[9px]" style={{ color: "#666" }}>NAIVE</span>
          <span className="text-[11px] font-medium" style={{ color: "#aaa" }}>
            +{naive?.toFixed(3)}s
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-[9px]" style={{ color: "#bbb" }}>INCHIDENT</span>
          <span className="text-[11px] font-semibold" style={{ color: "#eee" }}>
            +{clean?.toFixed(3)}s
          </span>
        </div>
        {delta && (
          <div
            className="flex items-center justify-between gap-4 mt-1 pt-1"
            style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}
          >
            <span className="text-[9px]" style={{ color: "#555" }}>Δ DELTA</span>
            <span className="text-[11px] font-semibold" style={{ color: "#fff" }}>
              {delta}s
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar() {
  const [dataFeedOpen, setDataFeedOpen] = useState(true);
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [modeOpen, setModeOpen] = useState(true);
  const [exportOpen, setExportOpen] = useState(true);

  const [fastf1, setFastf1] = useState(true);
  const [weather, setWeather] = useState(false);

  const [ers, setErs] = useState(true);
  const [traffic, setTraffic] = useState(true);
  const [liftcoast, setLiftcoast] = useState(false);
  const [thermal, setThermal] = useState(true);

  const [mode, setMode] = useState<"macro" | "micro">("macro");

  return (
    <aside
      className="flex flex-col overflow-y-auto py-4 px-4"
      style={{
        width: 200,
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        flexShrink: 0,
      }}
    >
      {/* DATA FEED */}
      <div>
        <button
          onClick={() => setDataFeedOpen(!dataFeedOpen)}
          className="w-full flex items-center justify-between mb-2 cursor-pointer"
        >
          <SectionHeading>Data Feed</SectionHeading>
          <span className="text-[10px] ml-1" style={{ color: "var(--text-muted)" }}>
            {dataFeedOpen ? "▾" : "▸"}
          </span>
        </button>
        {dataFeedOpen && (
          <div className="flex flex-col gap-2 mb-4">
            {[
              { label: "FastF1", val: fastf1, set: setFastf1 },
              { label: "Weather", val: weather, set: setWeather },
            ].map(({ label, val, set }) => (
              <label key={label} className="flex items-center gap-2 cursor-pointer group">
                <span
                  className="flex items-center justify-center w-3.5 h-3.5 shrink-0"
                  style={{
                    border: `1px solid ${val ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.15)"}`,
                    background: val ? "rgba(255,255,255,0.15)" : "transparent",
                  }}
                  onClick={() => set(!val)}
                >
                  {val && <span className="text-[8px] text-white">✓</span>}
                </span>
                <span className="text-[11px] tracking-wide" style={{ color: val ? "var(--text-primary)" : "var(--text-secondary)", fontFamily: "var(--mono)" }}>
                  {label}
                </span>
              </label>
            ))}
          </div>
        )}
      </div>

      <Divider />

      {/* FILTERS */}
      <div>
        <button
          onClick={() => setFiltersOpen(!filtersOpen)}
          className="w-full flex items-center justify-between cursor-pointer"
        >
          <SectionHeading>Filters</SectionHeading>
          <span className="text-[10px] ml-1" style={{ color: "var(--text-muted)" }}>
            {filtersOpen ? "▾" : "▸"}
          </span>
        </button>
        {filtersOpen && (
          <div className="flex flex-col gap-2.5 mb-4">
            {[
              { label: "ERS 2026 Limits", val: ers, set: setErs },
              { label: "Traffic Isolation", val: traffic, set: setTraffic },
              { label: "Lift/Coast Classifier", val: liftcoast, set: setLiftcoast },
              { label: "Thermal Memory", val: thermal, set: setThermal },
            ].map(({ label, val, set }) => (
              <div key={label} className="flex items-center justify-between gap-2">
                <span className="text-[10px] leading-tight" style={{ color: val ? "var(--text-primary)" : "var(--text-secondary)", fontFamily: "var(--mono)", flex: 1 }}>
                  {label}
                </span>
                <button
                  onClick={() => set(!val)}
                  className="relative shrink-0 cursor-pointer"
                  style={{ width: 28, height: 14 }}
                >
                  <span
                    className="absolute inset-0 rounded-none transition-colors"
                    style={{
                      background: val ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.06)",
                      border: "1px solid rgba(255,255,255,0.15)",
                    }}
                  />
                  <span
                    className="absolute top-0.5 transition-all"
                    style={{
                      width: 10,
                      height: 10,
                      background: val ? "#fff" : "#444",
                      left: val ? 16 : 2,
                    }}
                  />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <Divider />

      {/* ANALYSIS MODE */}
      <div>
        <button
          onClick={() => setModeOpen(!modeOpen)}
          className="w-full flex items-center justify-between cursor-pointer"
        >
          <SectionHeading>Analysis Mode</SectionHeading>
          <span className="text-[10px] ml-1" style={{ color: "var(--text-muted)" }}>
            {modeOpen ? "▾" : "▸"}
          </span>
        </button>
        {modeOpen && (
          <div className="flex flex-col gap-2 mb-4">
            {(["Macro", "Micro Sector"] as const).map((m) => {
              const key = m === "Macro" ? "macro" : "micro";
              return (
                <label key={m} className="flex items-center gap-2 cursor-pointer">
                  <span
                    className="flex items-center justify-center w-3.5 h-3.5 shrink-0 rounded-full"
                    style={{
                      border: `1px solid ${mode === key ? "rgba(255,255,255,0.6)" : "rgba(255,255,255,0.15)"}`,
                      background: mode === key ? "rgba(255,255,255,0.15)" : "transparent",
                    }}
                    onClick={() => setMode(key as "macro" | "micro")}
                  >
                    {mode === key && (
                      <span className="block w-1.5 h-1.5 rounded-full" style={{ background: "#fff" }} />
                    )}
                  </span>
                  <span
                    className="text-[11px] tracking-wide"
                    style={{ color: mode === key ? "var(--text-primary)" : "var(--text-secondary)", fontFamily: "var(--mono)" }}
                  >
                    {m}
                  </span>
                </label>
              );
            })}
          </div>
        )}
      </div>

      <Divider />

      {/* EXPORT */}
      <div>
        <button
          onClick={() => setExportOpen(!exportOpen)}
          className="w-full flex items-center justify-between cursor-pointer"
        >
          <SectionHeading>Export</SectionHeading>
          <span className="text-[10px] ml-1" style={{ color: "var(--text-muted)" }}>
            {exportOpen ? "▾" : "▸"}
          </span>
        </button>
        {exportOpen && (
          <div className="flex gap-2">
            {["CSV", "JSON"].map((fmt) => (
              <button
                key={fmt}
                className="flex-1 py-1.5 text-[10px] font-semibold tracking-widest transition-colors cursor-pointer"
                style={{
                  fontFamily: "var(--mono)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  color: "var(--text-secondary)",
                  background: "transparent",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.06)";
                  (e.currentTarget as HTMLButtonElement).style.color = "#eee";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.background = "transparent";
                  (e.currentTarget as HTMLButtonElement).style.color = "var(--text-secondary)";
                }}
              >
                {fmt}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1" />

      {/* Status pill */}
      <div
        className="mt-6 py-2 px-3 text-center text-[9px] tracking-widest flex items-center justify-center gap-2"
        style={{
          border: "1px solid var(--red-mid)",
          background: "var(--red-dim)",
          fontFamily: "var(--mono)",
          color: "rgba(232,80,80,0.9)",
        }}
      >
        <span className="inline-block w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: "var(--red)" }} />
        LIVE · 14:22:08 UTC
      </div>
    </aside>
  );
}

// ─── KPI Cards ────────────────────────────────────────────────────────────────

function KpiCard({
  title,
  primary,
  secondary,
  inverted = false,
  red = false,
  statusTag,
}: {
  title: string;
  primary: string;
  secondary: string;
  inverted?: boolean;
  red?: boolean;
  statusTag?: string;
}) {
  const bg = red
    ? "var(--red)"
    : inverted
    ? "rgba(255,255,255,0.95)"
    : "var(--panel)";
  const borderStyle = red
    ? "none"
    : inverted
    ? "none"
    : "1px solid var(--border)";

  return (
    <div
      className="flex flex-col justify-between p-4 relative overflow-hidden"
      style={{ background: bg, border: borderStyle, flex: 1 }}
    >
      {/* Corner accent */}
      <span
        className="absolute top-0 right-0 w-6 h-6"
        style={{
          borderLeft: `1px solid ${red ? "rgba(255,255,255,0.2)" : inverted ? "rgba(0,0,0,0.12)" : "var(--border-mid)"}`,
          borderBottom: `1px solid ${red ? "rgba(255,255,255,0.2)" : inverted ? "rgba(0,0,0,0.12)" : "var(--border-mid)"}`,
        }}
      />
      {/* Red: subtle noise texture via repeating gradient */}
      {red && (
        <span
          className="absolute inset-0 pointer-events-none"
          style={{
            background: "repeating-linear-gradient(135deg, rgba(255,255,255,0.03) 0px, rgba(255,255,255,0.03) 1px, transparent 1px, transparent 6px)",
          }}
        />
      )}

      <div className="flex items-start justify-between relative">
        <Label className={red ? "!text-[rgba(255,255,255,0.65)]" : inverted ? "!text-[#555]" : ""}>{title}</Label>
        {statusTag && (
          <span
            className="text-[9px] px-1.5 py-0.5 tracking-widest font-semibold"
            style={{
              fontFamily: "var(--mono)",
              background: red ? "rgba(0,0,0,0.18)" : inverted ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.06)",
              color: red ? "rgba(255,255,255,0.9)" : inverted ? "#222" : "#888",
            }}
          >
            {statusTag}
          </span>
        )}
      </div>

      <div className="relative">
        <div
          className="text-4xl font-bold leading-none tracking-tight mt-2 mb-1"
          style={{
            fontFamily: "var(--condensed)",
            color: red ? "#fff" : inverted ? "#0a0a0a" : "var(--text-primary)",
            letterSpacing: "-0.01em",
          }}
        >
          {primary}
        </div>
        <div
          className="text-[11px] tracking-wide"
          style={{
            fontFamily: "var(--mono)",
            color: red ? "rgba(255,255,255,0.65)" : inverted ? "#666" : "var(--text-secondary)",
          }}
        >
          {secondary}
        </div>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const dropdowns = [
    { label: "RACE", value: "Bahrain '26" },
    { label: "SESSION", value: "FP2" },
    { label: "DRIVER", value: "VER" },
    { label: "COMPOUND", value: "C3" },
  ];

  return (
    <div
      className="flex flex-col"
      style={{ height: "100%", background: "var(--bg)", overflow: "hidden" }}
    >
      {/* ── Header ── */}
      <header
        className="flex items-center justify-between px-5 shrink-0"
        style={{
          height: 48,
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div
            className="flex items-center justify-center"
            style={{
              width: 24,
              height: 24,
              border: "1px solid var(--red-mid)",
              background: "var(--red-dim)",
            }}
          >
            <span
              className="text-[10px] font-black"
              style={{ fontFamily: "var(--condensed)", color: "var(--red)", letterSpacing: "-0.03em" }}
            >
              T
            </span>
          </div>
          <div>
            <span
              className="text-[15px] font-bold tracking-[0.06em]"
              style={{ fontFamily: "var(--condensed)", color: "#fff" }}
            >
              INCHIDENT
            </span>
            <span
              className="text-[11px] ml-2 tracking-wide"
              style={{ fontFamily: "var(--condensed)", color: "var(--text-muted)", fontWeight: 400 }}
            >
              AI Motorsport Intelligence
            </span>
          </div>
        </div>

        {/* Dropdowns + gear */}
        <div className="flex items-center gap-2">
          {dropdowns.map(({ label, value }) => (
            <div
              key={label}
              className="flex items-center gap-1.5 px-2.5 py-1 cursor-pointer select-none"
              style={{
                border: "1px solid var(--border)",
                background: "var(--panel)",
                minWidth: 80,
              }}
            >
              <div className="flex flex-col leading-none">
                <span className="text-[8px] tracking-widest" style={{ fontFamily: "var(--mono)", color: "var(--text-muted)" }}>
                  {label}
                </span>
                <span className="text-[11px] font-semibold tracking-wide mt-0.5" style={{ fontFamily: "var(--mono)", color: "var(--text-primary)" }}>
                  {value}
                </span>
              </div>
              <span className="ml-auto text-[8px]" style={{ color: "var(--text-muted)" }}>▾</span>
            </div>
          ))}

          <button
            className="flex items-center justify-center cursor-pointer"
            style={{ width: 32, height: 32, border: "1px solid var(--border)", background: "var(--panel)" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </button>
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />

        {/* ── Main Content ── */}
        <main className="flex-1 flex flex-col overflow-hidden p-3 gap-3">
          {/* Row 1: KPI Cards */}
          <div className="flex gap-3 shrink-0" style={{ height: 120 }}>
            <KpiCard
              title="TYRE PENALTY"
              primary="0.61s/lap"
              secondary="Fresh Pace: 91.21s"
            />
            <KpiCard
              title="DEGRADATION"
              primary="0.034s/lap"
              secondary="Predicted Cliff: Lap 22"
            />
            <KpiCard
              title="UNDERCUT VALUE"
              primary="ACTIVE · GO"
              secondary="Net Gain: +1.2s"
              inverted
              statusTag="EXEC"
              red
            />
          </div>

          {/* Row 2: Main Chart */}
          <div
            className="flex flex-col shrink-0"
            style={{
              flex: "1 1 0",
              background: "var(--panel)",
              border: "1px solid var(--border)",
              padding: "14px 16px 10px",
              minHeight: 0,
            }}
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <Label>Primary Degradation Curve</Label>
                <div
                  className="text-[10px] mt-0.5"
                  style={{ fontFamily: "var(--mono)", color: "var(--text-muted)" }}
                >
                  C3 · Bahrain · FP2 · VER #1
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                  <svg width="22" height="6" viewBox="0 0 22 6">
                    <line x1="0" y1="3" x2="22" y2="3" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" strokeDasharray="3,2" />
                  </svg>
                  <Label>Naive Model</Label>
                </div>
                <div className="flex items-center gap-1.5">
                  <svg width="22" height="6" viewBox="0 0 22 6">
                    <line x1="0" y1="3" x2="22" y2="3" stroke="rgba(255,255,255,0.85)" strokeWidth="2.5" />
                  </svg>
                  <Label>Inchident Clean Pace</Label>
                </div>
              </div>
            </div>

            <div className="flex-1 min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={degData} margin={{ top: 4, right: 12, left: -8, bottom: 0 }}>
                  <CartesianGrid
                    strokeDasharray="0"
                    stroke="rgba(255,255,255,0.04)"
                    horizontal
                    vertical={false}
                  />
                  <XAxis
                    dataKey="lap"
                    tick={{ fontSize: 9, fill: "rgba(255,255,255,0.3)", fontFamily: "JetBrains Mono" }}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                    label={{
                      value: "TYRE AGE (LAP)",
                      position: "insideBottom",
                      offset: -2,
                      style: { fontSize: 8, fill: "rgba(255,255,255,0.2)", fontFamily: "JetBrains Mono", letterSpacing: "0.12em" },
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: "rgba(255,255,255,0.3)", fontFamily: "JetBrains Mono" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `+${v.toFixed(2)}s`}
                    label={{
                      value: "PACE DEFICIT",
                      angle: -90,
                      position: "insideLeft",
                      offset: 16,
                      style: { fontSize: 8, fill: "rgba(255,255,255,0.2)", fontFamily: "JetBrains Mono", letterSpacing: "0.12em" },
                    }}
                  />
                  <Tooltip content={<DegTooltip />} cursor={{ stroke: "rgba(255,255,255,0.1)", strokeWidth: 1 }} />
                  <ReferenceLine
                    x={22}
                    stroke="rgba(232,0,45,0.6)"
                    strokeDasharray="4 3"
                    strokeWidth={1.5}
                    label={{
                      value: "CLIFF",
                      position: "top",
                      style: { fontSize: 8, fill: "rgba(232,0,45,0.8)", fontFamily: "JetBrains Mono" },
                    }}
                  />
                  <Line
                    type="linear"
                    dataKey="naive"
                    stroke="rgba(255,255,255,0.22)"
                    strokeWidth={1}
                    dot={false}
                    strokeDasharray="3 2"
                  />
                  <Line
                    type="monotone"
                    dataKey="clean"
                    stroke="rgba(255,255,255,0.88)"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 3, fill: "#fff", stroke: "rgba(255,255,255,0.3)", strokeWidth: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Row 3: Split panels */}
          <div className="flex gap-3 shrink-0" style={{ height: 200 }}>
            {/* Panel 1: Scatter */}
            <div
              className="flex-1 flex flex-col"
              style={{
                background: "var(--panel)",
                border: "1px solid var(--border)",
                padding: "12px 14px 10px",
                minWidth: 0,
              }}
            >
              <div className="flex items-start justify-between mb-2 shrink-0">
                <Label>Micro-Sector Apex Analysis</Label>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1">
                    <span className="inline-block w-2 h-2 rounded-full" style={{ background: "rgba(255,255,255,0.25)" }} />
                    <Label>Normal</Label>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rotate-45" style={{ background: "rgba(232,0,45,0.12)", border: "1.5px solid #e8002d" }} />
                    <Label className="!text-[rgba(232,80,80,0.8)]">Traction Loss</Label>
                  </div>
                </div>
              </div>
              <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 4, right: 10, left: -12, bottom: 0 }}>
                    <CartesianGrid stroke="rgba(255,255,255,0.04)" />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[16, 26]}
                      tick={{ fontSize: 8, fill: "rgba(255,255,255,0.25)", fontFamily: "JetBrains Mono" }}
                      tickLine={false}
                      axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                      label={{
                        value: "THROTTLE DIST (m)",
                        position: "insideBottom",
                        offset: -2,
                        style: { fontSize: 7, fill: "rgba(255,255,255,0.18)", fontFamily: "JetBrains Mono", letterSpacing: "0.1em" },
                      }}
                    />
                    <YAxis
                      dataKey="y"
                      type="number"
                      domain={[100, 160]}
                      tick={{ fontSize: 8, fill: "rgba(255,255,255,0.25)", fontFamily: "JetBrains Mono" }}
                      tickLine={false}
                      axisLine={false}
                      label={{
                        value: "V_MIN (km/h)",
                        angle: -90,
                        position: "insideLeft",
                        offset: 16,
                        style: { fontSize: 7, fill: "rgba(255,255,255,0.18)", fontFamily: "JetBrains Mono", letterSpacing: "0.1em" },
                      }}
                    />
                    <Scatter
                      data={scatterNormal}
                      fill="rgba(255,255,255,0.22)"
                      shape={(props: any) => {
                        const { cx, cy } = props;
                        return <circle cx={cx} cy={cy} r={3} fill="rgba(255,255,255,0.2)" stroke="rgba(255,255,255,0.35)" strokeWidth={0.5} />;
                      }}
                    />
                    <Scatter
                      data={scatterOutlier}
                      fill="transparent"
                      shape={(props: any) => {
                        const { cx, cy } = props;
                        const s = 5;
                        return (
                          <g>
                            <rect
                              x={cx - s}
                              y={cy - s}
                              width={s * 2}
                              height={s * 2}
                              fill="rgba(232,0,45,0.12)"
                              stroke="#e8002d"
                              strokeWidth={1.5}
                              transform={`rotate(45 ${cx} ${cy})`}
                            />
                          </g>
                        );
                      }}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div
                className="shrink-0 mt-1 text-[8px] tracking-widest text-center"
                style={{ fontFamily: "var(--mono)", color: "rgba(232,0,45,0.7)" }}
              >
                ◆ TRACTION LOSS DETECTED · SECTOR 3 · T14-T16
              </div>
            </div>

            {/* Panel 2: Thermal & Energy */}
            <div
              className="flex-1 flex flex-col"
              style={{
                background: "var(--panel)",
                border: "1px solid var(--border)",
                padding: "12px 16px 10px",
                minWidth: 0,
              }}
            >
              <Label className="mb-3 block shrink-0">Thermal &amp; Energy State</Label>

              <div className="flex flex-col gap-4 flex-1">
                {/* Hysteresis */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <Label>Hysteresis (Thermal Abuse)</Label>
                    <span className="text-[10px]" style={{ fontFamily: "var(--mono)", color: "rgba(255,255,255,0.55)" }}>72%</span>
                  </div>
                  <div
                    className="relative h-3 w-full"
                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}
                  >
                    <div
                      className="h-full"
                      style={{
                        width: "72%",
                        background: "linear-gradient(90deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.45) 100%)",
                      }}
                    />
                    {/* Tick markers */}
                    {[25, 50, 75].map((t) => (
                      <div
                        key={t}
                        className="absolute top-0 bottom-0 w-px"
                        style={{ left: `${t}%`, background: "rgba(255,255,255,0.1)" }}
                      />
                    ))}
                  </div>
                  <div className="flex justify-between mt-1">
                    {["0", "25", "50", "75", "100"].map((v) => (
                      <span key={v} className="text-[8px]" style={{ fontFamily: "var(--mono)", color: "rgba(255,255,255,0.2)" }}>
                        {v}
                      </span>
                    ))}
                  </div>
                </div>

                {/* SOC Proxy */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <Label>SOC Proxy</Label>
                    <span className="text-[10px]" style={{ fontFamily: "var(--mono)", color: "rgba(255,255,255,0.55)" }}>38%</span>
                  </div>
                  <div
                    className="relative h-3 w-full"
                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}
                  >
                    <div
                      className="h-full"
                      style={{
                        width: "38%",
                        background: "linear-gradient(90deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.3) 100%)",
                      }}
                    />
                    {[25, 50, 75].map((t) => (
                      <div
                        key={t}
                        className="absolute top-0 bottom-0 w-px"
                        style={{ left: `${t}%`, background: "rgba(255,255,255,0.1)" }}
                      />
                    ))}
                  </div>
                  <div className="flex justify-between mt-1">
                    {["0", "25", "50", "75", "100"].map((v) => (
                      <span key={v} className="text-[8px]" style={{ fontFamily: "var(--mono)", color: "rgba(255,255,255,0.2)" }}>
                        {v}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Alert banner */}
              <div
                className="mt-3 shrink-0 flex items-center gap-2 px-3 py-2"
                style={{
                  background: "rgba(232,0,45,0.07)",
                  border: "1px solid rgba(232,0,45,0.25)",
                  borderLeft: "3px solid var(--red)",
                }}
              >
                <span className="text-[10px]" style={{ fontFamily: "var(--mono)", color: "var(--red)" }}>
                  ▲
                </span>
                <span
                  className="text-[10px] tracking-wider"
                  style={{ fontFamily: "var(--mono)", color: "rgba(232,80,80,0.95)" }}
                >
                  Anomaly: Lap 18 Lift &amp; Coast Flagged
                </span>
                <span
                  className="ml-auto text-[8px] tracking-widest shrink-0"
                  style={{ fontFamily: "var(--mono)", color: "rgba(232,0,45,0.4)" }}
                >
                  ERS-2026
                </span>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
