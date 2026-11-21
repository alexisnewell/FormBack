import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import axios from "axios";
import {
  getAnalysisResults,
  startAnalysis,
  uploadVideo,
  type UploadResponse,
} from "./api";
import "./App.css";

/* ------------------------------------------------------------------ */
/* Types (mirror api/schemas/analysis.py)                              */
/* ------------------------------------------------------------------ */

type Confidence = "high" | "medium" | "low" | "unvalidated";

interface MetricStat {
  label: string;
  unit?: string | null;
  median: number;
  iqr: number;
  n: number;
  confidence?: Confidence | null;
  confidence_note?: string | null;
}

interface Results {
  video_id: string;
  status: "processing" | "completed" | "failed";
  analysis_status?: "ok" | "low_quality";
  quality?: {
    score: number;
    label: string;
    landmark_coverage?: number | null;
    valid_stride_fraction?: number | null;
    stride_time_cv?: number | null;
    frames?: number | null;
  };
  warnings?: string[];
  analysis?: {
    fps?: number | null;
    direction?: string | null;
    estimated_cycle_period_s?: number | null;
  };
  metrics?: {
    overall: {
      cadence_spm?: number | null;
      symmetry_index_pct: Record<string, number | null>;
      foot_strike_pattern?: string | null;
    };
    per_side: Record<string, Record<string, MetricStat>>;
  };
  diagnostics?: {
    stride_counts: Record<string, number>;
  };
  error?: string | null;
}

type Phase =
  | { name: "idle" }
  | { name: "uploading"; file: File }
  | { name: "analyzing"; upload: UploadResponse }
  | { name: "done"; upload: UploadResponse; results: Results }
  | { name: "error"; message: string };

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const MAX_MB = 500;
const POLL_MS = 1500;

const METRIC_ORDER = [
  "stride_time_s",
  "contact_time_s",
  "flight_time_s",
  "duty_factor",
  "knee_flexion_at_contact_deg",
  "peak_stance_knee_flexion_deg",
  "ankle_ahead_of_pelvis_leg_lengths",
  "mean_trunk_lean_deg",
  "vertical_oscillation_pct_leg",
  "foot_strike_angle_deg",
];

function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (err.code === "ECONNABORTED") return "The request timed out.";
    if (!err.response)
      return "Can't reach the FormBack API. Is it running on localhost:8000?";
    return `Request failed (${err.response.status}).`;
  }
  return err instanceof Error ? err.message : "Something went wrong.";
}

function formatBytes(n: number): string {
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

function formatNumber(value: number, unit?: string | null): string {
  const digits = unit === "s" ? 3 : Math.abs(value) < 10 ? 2 : 1;
  return value.toFixed(digits);
}

function prettify(key: string): string {
  const s = key.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function symmetryTone(v: number | null | undefined): string {
  if (v == null) return "muted";
  const a = Math.abs(v);
  return a < 5 ? "good" : a < 10 ? "warn" : "bad";
}

/* ------------------------------------------------------------------ */
/* Insights engine                                                     */
/* ------------------------------------------------------------------ */

type Tone = "good" | "info" | "warn";

interface Insight {
  id: string;
  tone: Tone;
  title: string;
  detail: string;
  facts?: string[];
}

type Metrics = NonNullable<Results["metrics"]>;

const TONE_ORDER: Record<Tone, number> = { warn: 0, info: 1, good: 2 };
const TONE_GLYPH: Record<Tone, string> = { good: "✓", info: "i", warn: "!" };
const CAUTION =
  " This metric has lower measurement confidence, so treat it as a rough guide.";

function buildInsights(
  m: Metrics,
  diag: Results["diagnostics"],
): Insight[] {
  const out: Insight[] = [];
  const sides = Object.keys(m.per_side);

  const stats = (k: string) =>
    sides.map((s) => m.per_side[s][k]).filter((x): x is MetricStat => !!x);
  const mean = (k: string) => {
    const v = stats(k);
    return v.length ? v.reduce((a, x) => a + x.median, 0) / v.length : null;
  };
  const shaky = (k: string) => {
    const v = stats(k);
    return (
      v.length > 0 &&
      v.every((x) => x.confidence === "low" || x.confidence === "unvalidated")
    );
  };
  const label = (k: string) =>
    k === "step_time" ? "Step time" : (stats(k)[0]?.label ?? prettify(k));

  // Adds an insight; if the underlying metric is low-confidence, a "warn"
  // is softened to "info" and a caution is appended.
  const add = (i: Insight, basedOn?: string) => {
    if (basedOn && shaky(basedOn)) {
      out.push({
        ...i,
        tone: i.tone === "warn" ? "info" : i.tone,
        detail: i.detail + CAUTION,
      });
    } else {
      out.push(i);
    }
  };

  /* Data sufficiency */
  const counts = Object.values(diag?.stride_counts ?? {});
  if (counts.length && Math.min(...counts) < 10) {
    out.push({
      id: "few-strides",
      tone: "warn",
      title: "Short sample",
      detail:
        "Fewer than 10 strides were measured on at least one leg, so the numbers can swing a lot. A longer clip (20+ strides per leg) gives a steadier picture.",
      facts: sides.map((s) => `${prettify(s)}: ${diag?.stride_counts[s] ?? 0} strides`),
    });
  }

  /* Cadence */
  const cadence = m.overall.cadence_spm;
  if (cadence != null) {
    const c = Math.round(cadence);
    if (cadence < 150) {
      add({
        id: "cadence",
        tone: "info",
        title: "Cadence is on the lower side",
        detail:
          "Recreational runners average about 165 steps per minute at easy paces, and cadence rises with speed, so a low number at a slow pace is common. A modest increase (about 5%) shortens each stride and often reduces braking.",
        facts: [`${c} spm`],
      });
    } else if (cadence <= 185) {
      add({
        id: "cadence",
        tone: "good",
        title: "Cadence is in a typical range",
        detail:
          "Most recreational runners land between about 155 and 175 at easy paces, higher when faster. Compare between clips at a similar pace.",
        facts: [`${c} spm`],
      });
    } else {
      add({
        id: "cadence",
        tone: "info",
        title: "Cadence is high",
        detail:
          "A high step rate is common at fast paces and for shorter runners. If this was an easy run, check that the detected steps are correct.",
        facts: [`${c} spm`],
      });
    }
  }

  /* Symmetry */
  const sym = Object.entries(m.overall.symmetry_index_pct)
    .filter((e): e is [string, number] => e[1] != null)
    .map(([k, v]) => ({ k, v: Math.abs(v) }))
    .sort((a, b) => b.v - a.v);
  if (sym.length) {
    const big = sym.filter((x) => x.v >= 10);
    const mild = sym.filter((x) => x.v >= 5 && x.v < 10);
    const facts = (list: typeof sym) =>
      list.slice(0, 4).map((x) => `${label(x.k)}: ${x.v.toFixed(1)}%`);
    if (big.length) {
      add(
        {
          id: "symmetry",
          tone: "warn",
          title: `Left and right differ in ${big.length === 1 ? label(big[0].k).toLowerCase() : `${big.length} metrics`}`,
          detail:
          "Some difference between legs is normal, and in a large study of recreational runners asymmetry wasn't linked to injury risk. Video measurement adds its own error, so film a second clip before reading much into it.",
          facts: facts(big),
        },
        big[0].k !== "step_time" ? big[0].k : undefined,
      );
    } else if (mild.length) {
      add({
        id: "symmetry",
        tone: "info",
        title: "Mild left/right differences",
        detail:
          "A little asymmetry is normal. Nothing here is large enough to be a concern on its own.",
        facts: facts(mild),
      });
    } else {
      out.push({
        id: "symmetry",
        tone: "good",
        title: "Left and right look well balanced",
        detail: "Every measured metric differs by less than 5% between legs.",
        facts: [`Largest: ${label(sym[0].k)} ${sym[0].v.toFixed(1)}%`],
      });
    }
  }

  /* Foot strike + overstride */
  const pattern = m.overall.foot_strike_pattern;
  const angle = mean("foot_strike_angle_deg");
  const ahead = mean("ankle_ahead_of_pelvis_leg_lengths");
  if (pattern) {
    const text: Record<string, string> = {
      rearfoot:
        "Your heel tends to land first. This is the most common pattern and isn’t a problem by itself. What matters more is how far in front of your body the foot lands.",
      midfoot:
        "Your foot tends to land flat, which spreads impact across the foot.",
      forefoot:
        "You tend to land on the ball of the foot. This loads the calf and Achilles more, so build up gradually if you’re changing toward it.",
    };
    add(
      {
        id: "foot-strike",
        tone: "info",
        title: `${prettify(pattern)} strike`,
        detail: text[pattern] ?? "",
        facts: angle != null ? [`Foot angle at contact: ${angle.toFixed(1)}°`] : [],
      },
      "foot_strike_angle_deg",
    );
  } else {
    out.push({
      id: "foot-strike",
      tone: "info",
      title: "Foot strike varies from stride to stride",
      detail:
        "Your landing pattern wasn’t consistent enough to label. This is common and can also reflect video angle or frame rate.",
    });
  }

  if (ahead != null) {
    add(
      {
        id: "overstride",
        tone: "info",
        title: "Foot placement relative to your hips",
        detail:
          "This is how far the ankle is ahead of the pelvis at contact, in leg lengths. Higher values mean the foot lands further in front of you. There's no agreed cutoff for overstriding, so track it across clips rather than judging a single number.",
        facts: [`${ahead.toFixed(2)} leg lengths ahead`],
      },
      "ankle_ahead_of_pelvis_leg_lengths",
    );
  }

  /* Trunk lean */
  const lean = mean("mean_trunk_lean_deg");
  if (lean != null) {
    const l = Math.abs(lean);
    if (l < 2) {
      add(
        {
          id: "lean",
          tone: "info",
          title: "Very upright posture",
          detail:
            "A larger lean shifts load from the knee toward the hip and can raise energy cost. Check whether it comes from the hips or from the whole body tilting, and whether it eases at a slower pace.",
          facts: [`${lean.toFixed(1)}° trunk lean`],
        },
        "mean_trunk_lean_deg",
      );
    } else if (l <= 12) {
      add(
        {
          id: "lean",
          tone: "good",
          title: "Natural forward lean",
          detail:
            "A small forward lean like this is typical and generally efficient.",
          facts: [`${lean.toFixed(1)}° trunk lean`],
        },
        "mean_trunk_lean_deg",
      );
    } else {
      add(
        {
          id: "lean",
          tone: "warn",
          title: "Pronounced forward lean",
          detail:
            "Leaning this far can load the lower back and hip flexors. Check whether it’s from the hips (bending at the waist) or from the whole body tilting, and whether it eases at a slower pace.",
          facts: [`${lean.toFixed(1)}° trunk lean`],
        },
        "mean_trunk_lean_deg",
      );
    }
  }

  /* Vertical oscillation */
  const osc = mean("vertical_oscillation_pct_leg");
  if (osc != null) {
    if (osc <= 10) {
      add(
        {
          id: "bounce",
          tone: "good",
          title: "Smooth vertical motion",
          detail:
            "Your body moves up and down relatively little, so more of your effort goes forward.",
          facts: [`${osc.toFixed(1)}% of leg length`],
        },
        "vertical_oscillation_pct_leg",
      );
    } else if (osc <= 13) {
      add(
        {
          id: "bounce",
          tone: "info",
          title: "Moderate vertical bounce",
          detail:
            "A bit of bounce is normal, and it increases with speed. Staying low and quick off the ground can trim it.",
          facts: [`${osc.toFixed(1)}% of leg length`],
        },
        "vertical_oscillation_pct_leg",
      );
    } else {
      add(
        {
          id: "bounce",
          tone: "warn",
          title: "High vertical bounce",
          detail:
            "You’re spending noticeable energy moving up rather than forward. Higher cadence and a lighter, quicker push-off usually reduce this.",
          facts: [`${osc.toFixed(1)}% of leg length`],
        },
        "vertical_oscillation_pct_leg",
      );
    }
  }

  /* Ground contact time */
  const gct = mean("contact_time_s");
  if (gct != null) {
    const ms = Math.round(gct * 1000);
    if (gct > 0.3) {
      add(
        {
          id: "contact",
          tone: "info",
          title: "Longer ground contact",
          detail:
            "Contact times above about 300 ms are common at easy paces. Faster running and a higher cadence tend to shorten them.",
          facts: [`${ms} ms`],
        },
        "contact_time_s",
      );
    } else {
      add(
        {
          id: "contact",
          tone: "good",
          title: "Quick ground contact",
          detail: "Your feet leave the ground promptly, which is a good sign for efficiency.",
          facts: [`${ms} ms`],
        },
        "contact_time_s",
      );
    }
  }

  return out.sort((a, b) => TONE_ORDER[a.tone] - TONE_ORDER[b.tone]);
}

/* ------------------------------------------------------------------ */
/* Small components                                                    */
/* ------------------------------------------------------------------ */

export function Logo({ size = 40 }: { size?: number }) {
  return (
    <svg
      className="fb-logo"
      width={size}
      height={size}
      viewBox="0 0 48 48"
      role="img"
      aria-label="FormBack logo"
    >
      <defs>
        <linearGradient id="fb-logo-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#c084fc" />
          <stop offset="1" stopColor="#6d28d9" />
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="13" fill="url(#fb-logo-bg)" />
      {/* feedback arc trailing behind the runner */}
      <path
        d="M9 21 A15 15 0 0 1 20 9"
        fill="none"
        stroke="#fff"
        strokeOpacity="0.45"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      <path
        d="M9 27 A15 15 0 0 0 13 34"
        fill="none"
        stroke="#fff"
        strokeOpacity="0.25"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
      {/* runner */}
      <g
        fill="none"
        stroke="#fff"
        strokeWidth="3.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M27 17 L23.5 27" />
        <path d="M26 19 L32 22.5 L36 20.5" />
        <path d="M25.5 19.5 L19.5 21 L17 25.5" />
        <path d="M23.5 27 L31 30.5 L30 38" />
        <path d="M23.5 27 L19.5 33.5 L12.5 35.5" />
      </g>
      <circle cx="29.5" cy="11.5" r="3.6" fill="#fff" />
    </svg>
  );
}

function MetricsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M4 20V10" />
      <path d="M10 20V4" />
      <path d="M16 20v-7" />
      <path d="M22 20H2" />
    </svg>
  );
}

function InsightsIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 18h6" />
      <path d="M10 21h4" />
      <path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z" />
    </svg>
  );
}

function Spinner() {
  return <span className="fb-spinner" aria-hidden="true" />;
}

function ConfidenceBadge({ stat }: { stat: MetricStat }) {
  if (!stat.confidence) return null;
  return (
    <span
      className={`fb-badge fb-badge-${stat.confidence}`}
      title={stat.confidence_note ?? `${stat.confidence} confidence`}
    >
      {stat.confidence}
    </span>
  );
}

function UploadCard({
  onFile,
  busy,
  error,
}: {
  onFile: (file: File) => void;
  busy: boolean;
  error?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const pick = (file: File | undefined) => {
    if (file) onFile(file);
  };

  return (
    <div
      className={`fb-drop ${dragging ? "is-dragging" : ""}`}
      onDragOver={(e: DragEvent) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e: DragEvent) => {
        e.preventDefault();
        setDragging(false);
        pick(e.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,.mp4"
        hidden
        onChange={(e: ChangeEvent<HTMLInputElement>) => {
          pick(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      <h2>Upload a running video</h2>
      <p>Drop an MP4 here, or choose a file. Up to {MAX_MB} MB.</p>
      <button
        className="fb-btn fb-btn-primary"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
      >
        Choose video
      </button>
      {error && (
        <p className="fb-error" role="alert">
          {error}
        </p>
      )}
      <ul className="fb-tips">
        <li>Film from the side, with the full body in frame.</li>
        <li>Keep the camera steady and film at least 10 strides.</li>
        <li>Good lighting and a plain background improve tracking.</li>
      </ul>
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: string;
}) {
  return (
    <div className={`fb-stat ${tone ? `tone-${tone}` : ""}`}>
      <span className="fb-stat-label">{label}</span>
      <span className="fb-stat-value">{value}</span>
      {sub && <span className="fb-stat-sub">{sub}</span>}
    </div>
  );
}

function InsightsView({ insights }: { insights: Insight[] }) {
  const count = (t: Tone) => insights.filter((i) => i.tone === t).length;

  if (!insights.length) {
    return (
      <section className="fb-card">
        <h2>Insights</h2>
        <p className="fb-muted">Not enough data to draw insights from this clip.</p>
      </section>
    );
  }

  return (
    <section className="fb-card">
      <h2>Insights</h2>
      <p className="fb-insight-summary">
        <span className="fb-pill warn">{count("warn")} worth a look</span>
        <span className="fb-pill info">{count("info")} to note</span>
        <span className="fb-pill good">{count("good")} looking good</span>
      </p>
      <ul className="fb-insights">
        {insights.map((i) => (
          <li key={i.id} className={`fb-insight fb-insight-${i.tone}`}>
            <span className="fb-insight-glyph" aria-hidden="true">
              {TONE_GLYPH[i.tone]}
            </span>
            <div>
              <h3>{i.title}</h3>
              <p>{i.detail}</p>
              {i.facts && i.facts.length > 0 && (
                <p className="fb-facts">
                  {i.facts.map((f) => (
                    <span key={f} className="fb-fact">
                      {f}
                    </span>
                  ))}
                </p>
              )}
            </div>
          </li>
        ))}
      </ul>
      <p className="fb-muted fb-foot">
      Comparisons are based on published studies of recreational runners see read me for more info.
      </p>
    </section>
  );
}

function ResultsView({
  upload,
  results,
}: {
  upload: UploadResponse;
  results: Results;
}) {
    const { quality, metrics, analysis, diagnostics } = results;
    const warnings: string[] = [];
  const [tab, setTab] = useState<"metrics" | "insights">("metrics");
  const insights = metrics ? buildInsights(metrics, diagnostics) : [];
  const lowQuality = results.analysis_status === "low_quality";

  const sides = metrics ? Object.keys(metrics.per_side) : [];
  const keys = metrics
    ? Array.from(
        new Set(sides.flatMap((s) => Object.keys(metrics.per_side[s]))),
      ).sort((a, b) => {
        const ia = METRIC_ORDER.indexOf(a);
        const ib = METRIC_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
      })
    : [];

  const labelFor = (key: string) =>
    sides.map((s) => metrics?.per_side[s][key]?.label).find(Boolean) ??
    prettify(key);

  return (
    <div className="fb-results">
      {/* Quality */}
      {quality && (
        <section className={`fb-card ${lowQuality ? "fb-card-warn" : ""}`}>
          <div className="fb-card-head">
            <h2>{lowQuality ? "Low-quality analysis" : "Analysis quality"}</h2>
            <span className={`fb-pill ${lowQuality ? "bad" : "good"}`}>
              {quality.label}
            </span>
          </div>
          <div className="fb-meter" aria-label={`Quality score ${quality.score}`}>
            <div
              className={`fb-meter-fill ${lowQuality ? "bad" : "good"}`}
              style={{ width: `${Math.round(quality.score * 100)}%` }}
            />
          </div>
          <p className="fb-muted">
            Score {quality.score.toFixed(2)}
            {quality.landmark_coverage != null &&
              ` · landmarks tracked in ${Math.round(quality.landmark_coverage * 100)}% of frames`}
            {quality.valid_stride_fraction != null &&
              ` · ${Math.round(quality.valid_stride_fraction * 100)}% of strides valid`}
          </p>
          {lowQuality && (
            <p>
              The numbers from this clip aren’t reliable enough to show. Try a
              clip filmed from the side, with steady framing, good light, and the
              whole body visible.
            </p>
          )}
          {warnings.length > 0 && (
            <ul className="fb-warnings">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* Tabs */}
      {metrics && (
        <div className="fb-tabs" role="tablist" aria-label="Results">
          {(["metrics", "insights"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              id={`tab-${t}`}
              aria-selected={tab === t}
              aria-controls="fb-tabpanel"
              className={`fb-tab ${tab === t ? "is-active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t === "metrics" ? <MetricsIcon /> : <InsightsIcon />}
              {t === "metrics" ? "Metrics" : "Insights"}
              {t === "insights" && insights.length > 0 && (
                <span className="fb-tab-count">{insights.length}</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Overall */}
      {metrics && tab === "metrics" && (
        <section className="fb-card">
          <h2>Overall</h2>
          <div className="fb-stats">
            <Stat
              label="Cadence"
              value={
                metrics.overall.cadence_spm != null
                  ? `${Math.round(metrics.overall.cadence_spm)} spm`
                  : "—"
              }
            />
            <Stat
              label="Foot strike"
              value={
                metrics.overall.foot_strike_pattern
                  ? prettify(metrics.overall.foot_strike_pattern)
                  : "Not classified"
              }
              sub={
                metrics.overall.foot_strike_pattern
                  ? undefined
                  : "Too variable to classify"
              }
            />
            {analysis?.direction && (
              <Stat label="Running direction" value={prettify(analysis.direction)} />
            )}
            <Stat
              label="Clip"
              value={`${upload.duration_s.toFixed(1)} s`}
              sub={`${upload.fps.toFixed(0)} fps`}
            />
          </div>
        </section>
      )}

      {/* Per-side metrics */}
      {metrics && tab === "metrics" && keys.length > 0 && (
        <section className="fb-card">
          <div className="fb-card-head">
            <h2>Left vs. right</h2>
            <span className="fb-muted">
              median · IQR · strides
            </span>
          </div>
          <div className="fb-table-wrap">
            <table className="fb-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {sides.map((s) => (
                    <th key={s}>
                      {prettify(s)}
                      {diagnostics?.stride_counts[s] != null && (
                        <span className="fb-th-sub">
                          {diagnostics.stride_counts[s]} strides
                        </span>
                      )}
                    </th>
                  ))}
                  <th>Asymmetry</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((key) => {
                  const sym = metrics.overall.symmetry_index_pct[key];
                  return (
                    <tr key={key}>
                      <th scope="row">{labelFor(key)}</th>
                      {sides.map((s) => {
                        const m = metrics.per_side[s][key];
                        return (
                          <td key={s}>
                            {m ? (
                              <>
                                <span className="fb-val">
                                  {formatNumber(m.median, m.unit)}
                                  {m.unit && (
                                    <span className="fb-unit"> {m.unit}</span>
                                  )}
                                </span>
                                <span className="fb-cell-sub">
                                  IQR {formatNumber(m.iqr, m.unit)} · n={m.n}{" "}
                                  <ConfidenceBadge stat={m} />
                                </span>
                              </>
                            ) : (
                              <span className="fb-muted">—</span>
                            )}
                          </td>
                        );
                      })}
                      <td>
                        {sym != null ? (
                          <span className={`fb-pill ${symmetryTone(sym)}`}>
                            {Math.abs(sym).toFixed(1)}%
                          </span>
                        ) : (
                          <span className="fb-muted">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
                {metrics.overall.symmetry_index_pct.step_time != null && (
                  <tr>
                    <th scope="row">Step time</th>
                    <td colSpan={sides.length} className="fb-muted">
                      Left/right timing difference
                    </td>
                    <td>
                      <span
                        className={`fb-pill ${symmetryTone(
                          metrics.overall.symmetry_index_pct.step_time,
                        )}`}
                      >
                        {Math.abs(
                          metrics.overall.symmetry_index_pct.step_time,
                        ).toFixed(1)}
                        %
                      </span>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <p className="fb-muted fb-foot">
            Asymmetry of 0% is perfectly symmetric. Hover a confidence badge for
            details on why a metric may be less reliable.
          </p>
        </section>
      )}
      {/* Insights */}
      {metrics && tab === "insights" && <InsightsView insights={insights} />}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main page                                                           */
/* ------------------------------------------------------------------ */

export default function App() {
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [validation, setValidation] = useState<string>();
  const runId = useRef(0); // lets a reset cancel an in-flight poll loop

  // Revoke object URLs when replaced/unmounted.
  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  useEffect(() => () => void (runId.current += 1), []);

  const reset = useCallback(() => {
    runId.current += 1;
    setPhase({ name: "idle" });
    setVideoUrl(null);
    setValidation(undefined);
  }, []);

  const handleFile = useCallback(async (file: File) => {
    setValidation(undefined);

    if (!file.name.toLowerCase().endsWith(".mp4")) {
      setValidation("Only MP4 videos are supported.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setValidation(`That video is over the ${MAX_MB} MB limit.`);
      return;
    }

    const id = ++runId.current;
    const cancelled = () => id !== runId.current;

    setVideoUrl(URL.createObjectURL(file));
    setPhase({ name: "uploading", file });

    try {
      const upload = await uploadVideo(file);
      if (cancelled()) return;

      await startAnalysis(upload.video_id);
      if (cancelled()) return;
      setPhase({ name: "analyzing", upload });

      // Poll the results endpoint: 202 while processing, 200 once finished.
      for (;;) {
        await new Promise((r) => setTimeout(r, POLL_MS));
        if (cancelled()) return;

        const results = (await getAnalysisResults(upload.video_id)) as Results;
        if (cancelled()) return;

        if (results.status === "processing") continue;

        if (results.status === "failed") {
          setPhase({
            name: "error",
            message: results.error ?? "The analysis failed.",
          });
        } else {
          setPhase({ name: "done", upload, results });
        }
        return;
      }
    } catch (err) {
      if (cancelled()) return;
      setPhase({ name: "error", message: errorMessage(err) });
    }
  }, []);

  const busy = phase.name === "uploading" || phase.name === "analyzing";
  const upload =
    phase.name === "analyzing" || phase.name === "done" ? phase.upload : null;

  return (
    <div className="fb">
      <header className="fb-header">
        <div className="fb-brand">
          <Logo />
          <h1>
            Form<span className="fb-wordmark-accent">Back</span>
          </h1>
        </div>
        <p className="fb-tagline">
          Running form analysis from a single side-on video.
        </p>
      </header>

      <main className="fb-main">
        {phase.name === "idle" && (
          <UploadCard onFile={handleFile} busy={false} error={validation} />
        )}

        {phase.name === "error" && (
          <>
            <div className="fb-card fb-card-warn" role="alert">
              <h2>Something went wrong</h2>
              <p>{phase.message}</p>
              <button className="fb-btn fb-btn-primary" onClick={reset}>
                Try another video
              </button>
            </div>
          </>
        )}

        {phase.name !== "idle" && phase.name !== "error" && (
          <div className="fb-layout">
            <aside className="fb-side">
              <div className="fb-card">
                {videoUrl && (
                  <video
                    className="fb-video"
                    src={videoUrl}
                    controls
                    muted
                    playsInline
                  />
                )}
                <p className="fb-file">
                  {phase.name === "uploading"
                    ? phase.file.name
                    : upload?.filename ?? ""}
                </p>
                {upload && (
                  <p className="fb-muted">
                    {formatBytes(upload.size_bytes)} · {upload.duration_s.toFixed(1)} s ·{" "}
                    {upload.fps.toFixed(0)} fps · {upload.frame_count} frames
                  </p>
                )}
                <button
                  className="fb-btn"
                  onClick={reset}
                >
                  {busy ? "Cancel" : "Analyze another video"}
                </button>
              </div>
            </aside>

            <section className="fb-content">
              {busy && (
                <div className="fb-card fb-progress" aria-live="polite">
                  <Spinner />
                  <div>
                    <h2>
                      {phase.name === "uploading"
                        ? "Uploading video…"
                        : "Analyzing your stride…"}
                    </h2>
                    <p className="fb-muted">
                      {phase.name === "uploading"
                        ? "Large files can take a moment."
                        : "Detecting pose, finding foot strikes, and measuring each stride. This usually takes a little while."}
                    </p>
                  </div>
                </div>
              )}

              {phase.name === "done" && (
                <ResultsView upload={phase.upload} results={phase.results} />
              )}
            </section>
          </div>
        )}
      </main>

      <footer className="fb-footer">
        FormBack provides general movement insights, not medical advice.
      </footer>
    </div>
  );
}