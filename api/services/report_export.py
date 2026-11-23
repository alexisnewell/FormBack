"""
Report export: turns a finished ``ResultsResponse`` into a downloadable file.

    CSV   one tidy table of every metric (for spreadsheets / your own analysis)
    PDF   a readable report with the numbers and charts

Both are built from the same ``ResultsResponse`` the API returns, so they can
never show anything the results endpoint would not (the quality gate in
report_service already withheld metrics for low-quality analyses).

Charts:
    * a left-vs-right comparison drawn here with matplotlib's ``Figure`` object.
      That avoids pyplot's global state, so it is safe to call from any of
      FastAPI's request threads.
    * the stride-cycle and event-detection PNGs the pipeline already wrote.
"""

import csv
import io
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from api.schemas.analysis import MetricStat, ResultsResponse
from api.services.insights import Insight, build_insights

ACCENT = colors.HexColor("#6d28d9")
MUTED = colors.HexColor("#6b7280")
GRID = colors.HexColor("#e5e7eb")
TONE_COLORS = {
    "warn": colors.HexColor("#b45309"),
    "info": colors.HexColor("#4338ca"),
    "good": colors.HexColor("#15803d"),
}
TONE_LABELS = {"warn": "WORTH A LOOK", "info": "TO NOTE", "good": "LOOKING GOOD"}
LEFT_COLOR = "#7c3aed"
RIGHT_COLOR = "#f59e0b"

DISCLAIMER = "FormBack provides general movement insights, not medical advice."


# --------------------------------------------------
# SHARED HELPERS
# --------------------------------------------------

def _prettify(key: str) -> str:
    s = key.replace("_", " ")
    return s[:1].upper() + s[1:]


def _fmt(value: float, unit: Optional[str]) -> str:
    """Same rounding the web UI uses."""
    digits = 3 if unit == "s" else 2 if abs(value) < 10 else 1
    return f"{value:.{digits}f}"


def _metric_keys(results: ResultsResponse) -> List[str]:
    """Metric keys present on either leg, in the order the pipeline defines."""
    from api.services.report_service import METRICS  # lazy: avoids a cycle

    present = set()
    for stats in results.metrics.per_side.values():
        present.update(stats)
    return [k for k in METRICS if k in present] + sorted(present - set(METRICS))


def _sides(results: ResultsResponse) -> List[str]:
    return list(results.metrics.per_side)


# --------------------------------------------------
# CSV
# --------------------------------------------------

CSV_COLUMNS = [
    "section", "metric", "label", "side", "unit",
    "median", "iqr", "n", "confidence", "note",
]


def build_csv(results: ResultsResponse) -> bytes:
    """Long-format CSV: one row per value.

    section   overall | symmetry | per_side
    side      left / right for per_side rows, empty otherwise
    Written with a UTF-8 BOM so Excel opens it with the right encoding.
    """
    m = results.metrics
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()

    def row(**kw):
        w.writerow({c: kw.get(c, "") for c in CSV_COLUMNS})

    if m.overall.cadence_spm is not None:
        row(section="overall", metric="cadence_spm", label="Cadence",
            unit="spm", median=m.overall.cadence_spm)
    if m.overall.foot_strike_pattern:
        row(section="overall", metric="foot_strike_pattern",
            label="Foot strike pattern", note=m.overall.foot_strike_pattern)

    for key, value in m.overall.symmetry_index_pct.items():
        if value is None:
            continue
        label = "Step time" if key == "step_time" else _label_for(results, key)
        row(section="symmetry", metric=key, label=label, unit="%", median=value)

    for side, stats in m.per_side.items():
        for key in _metric_keys(results):
            s = stats.get(key)
            if s is None:
                continue
            row(section="per_side", metric=key, label=s.label, side=side,
                unit=s.unit or "", median=s.median, iqr=s.iqr, n=s.n,
                confidence=s.confidence.value if s.confidence else "",
                note=s.confidence_note or "")

    return buf.getvalue().encode("utf-8-sig")


def _label_for(results: ResultsResponse, key: str) -> str:
    for stats in results.metrics.per_side.values():
        if key in stats:
            return stats[key].label
    return _prettify(key)


# --------------------------------------------------
# CHARTS
# --------------------------------------------------

def build_comparison_chart(results: ResultsResponse) -> Optional[bytes]:
    """Small-multiples bar chart: one panel per metric, left vs right median."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sides = _sides(results)
    keys = _metric_keys(results)
    if not keys or not sides:
        return None

    cols = min(5, len(keys))
    rows = math.ceil(len(keys) / cols)
    fig = Figure(figsize=(2.3 * cols, 2.5 * rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, squeeze=False)
    palette = {"left": LEFT_COLOR, "right": RIGHT_COLOR}

    for ax, key in zip(axes.flat, keys):
        values, bar_colors, ticks = [], [], []
        stat: Optional[MetricStat] = None
        for side in sides:
            s = results.metrics.per_side[side].get(key)
            if s is None:
                continue
            stat = stat or s
            values.append(s.median)
            bar_colors.append(palette.get(side, "#9ca3af"))
            ticks.append(side[:1].upper())
        bars = ax.bar(range(len(values)), values, color=bar_colors, width=0.6)
        ax.set_xticks(range(len(values)), ticks)
        for bar, v in zip(bars, values):
            ax.annotate(_fmt(v, stat.unit if stat else None),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7,
                        xytext=(0, 2), textcoords="offset points")
        label = "\n".join(textwrap.wrap(stat.label if stat else _prettify(key), 20))
        unit = f" ({stat.unit})" if stat and stat.unit else ""
        ax.set_title(label + unit, fontsize=8)
        ax.margins(y=0.18)
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)

    fig.tight_layout()
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=150)
    return out.getvalue()


# --------------------------------------------------
# PDF
# --------------------------------------------------

def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], alignment=0,
                                textColor=ACCENT, fontSize=22, leading=26, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=base["Normal"], textColor=MUTED,
                              fontSize=9, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], textColor=ACCENT,
                             fontSize=13, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.5,
                               leading=13),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=8,
                                leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=9,
                               leading=11),
        "cellsub": ParagraphStyle("cs", parent=base["Normal"], fontSize=7.5,
                                  leading=9.5, textColor=MUTED),
    }


def _table(data, col_widths, header=True) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f3ff")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ]
    t.setStyle(TableStyle(style))
    return t


def _image(source, max_width: float, max_height: float) -> Image:
    """Scale an image to fit the box, keeping its aspect ratio."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    w, h = ImageReader(source).getSize()
    if hasattr(source, "seek"):
        source.seek(0)
    scale = min(max_width / w, max_height / h)
    return Image(source, width=w * scale, height=h * scale)


def _cell_for(stat: Optional[MetricStat], st) -> List:
    if stat is None:
        return [Paragraph("&mdash;", st["cell"])]
    unit = f" {escape(stat.unit)}" if stat.unit else ""
    conf = f" &middot; {stat.confidence.value}" if stat.confidence else ""
    return [
        Paragraph(f"{_fmt(stat.median, stat.unit)}{unit}", st["cell"]),
        Paragraph(f"IQR {_fmt(stat.iqr, stat.unit)} &middot; n={stat.n}{conf}",
                  st["cellsub"]),
    ]


def _insight_block(i: Insight, st) -> KeepTogether:
    """One insight: a tone tag (text, not colour alone), title, detail, facts."""
    color = TONE_COLORS[i.tone].hexval()[2:]
    items = [
        Paragraph(
            f'<font color="#{color}" size="7.5"><b>{TONE_LABELS[i.tone]}</b></font>',
            st["cell"]),
        Paragraph(f"<b>{escape(i.title)}</b>", st["body"]),
        Paragraph(escape(i.detail), st["body"]),
    ]
    if i.facts:
        items.append(Paragraph(" &middot; ".join(escape(f) for f in i.facts), st["small"]))
    items.append(Spacer(1, 8))
    return KeepTogether(items)


def build_pdf(
    results: ResultsResponse,
    video_id: str,
    chart_paths: Optional[Dict[str, Path]] = None,
) -> bytes:
    """Render the report. ``chart_paths`` maps artefact name -> existing PNG
    (``cycles_plot``, ``debug_plot``); missing entries are skipped."""
    st = _styles()
    m = results.metrics
    sides = _sides(results)
    chart_paths = chart_paths or {}
    content_w = A4[0] - 4 * cm
    story: list = []

    # Header
    story.append(Paragraph("FormBack running form report", st["title"]))
    story.append(Paragraph(
        f"Generated {datetime.now():%d %b %Y, %H:%M} &middot; "
        f"video {escape(video_id[:8])}", st["sub"]))

    # Quality + warnings
    q = results.quality
    if q:
        story.append(Paragraph("Analysis quality", st["h2"]))
        parts = [f"Score {q.score:.2f} ({escape(q.label)})"]
        if q.landmark_coverage is not None:
            parts.append(f"landmarks tracked in {round(q.landmark_coverage * 100)}% of frames")
        if q.valid_stride_fraction is not None:
            parts.append(f"{round(q.valid_stride_fraction * 100)}% of strides valid")
        story.append(Paragraph(" &middot; ".join(parts), st["body"]))
    if results.warnings:
        story.append(Spacer(1, 4))
        for w in results.warnings:
            story.append(Paragraph(f"&bull; {escape(w)}", st["body"]))

    # Overall
    story.append(Paragraph("Overall", st["h2"]))
    overall = [["Measure", "Value"]]
    if m.overall.cadence_spm is not None:
        overall.append(["Cadence", f"{round(m.overall.cadence_spm)} spm"])
    overall.append(["Foot strike", _prettify(m.overall.foot_strike_pattern)
                    if m.overall.foot_strike_pattern else "Not classified (too variable)"])
    if results.analysis:
        if results.analysis.direction:
            overall.append(["Running direction", _prettify(results.analysis.direction)])
        if results.analysis.fps:
            overall.append(["Video frame rate", f"{results.analysis.fps:.0f} fps"])
    if results.diagnostics:
        for side, n in results.diagnostics.stride_counts.items():
            overall.append([f"{_prettify(side)} strides measured", str(n)])
    story.append(_table(overall, [content_w * 0.5, content_w * 0.5]))

    # Insights (same observations as the Insights tab in the web app)
    insights = build_insights(m, results.diagnostics)
    story.append(Paragraph("Insights", st["h2"]))
    if insights:
        n = {t: sum(1 for i in insights if i.tone == t) for t in TONE_LABELS}
        story.append(Paragraph(
            f"{n['warn']} worth a look &middot; {n['info']} to note &middot; "
            f"{n['good']} looking good", st["small"]))
        story.append(Spacer(1, 6))
        story.extend(_insight_block(i, st) for i in insights)
        story.append(Paragraph(
            "Comparisons are based on published studies of recreational "
            "runners. These are general observations, not a diagnosis.",
            st["small"]))
    else:
        story.append(Paragraph(
            "Not enough data to draw insights from this clip.", st["body"]))

    # Left vs right table
    keys = _metric_keys(results)
    if keys:
        story.append(Paragraph("Left vs. right", st["h2"]))
        head = ["Metric"] + [_prettify(s) for s in sides] + ["Asymmetry"]
        rows = [head]
        for key in keys:
            sym = m.overall.symmetry_index_pct.get(key)
            rows.append(
                [Paragraph(escape(_label_for(results, key)), st["cell"])]
                + [_cell_for(m.per_side[s].get(key), st) for s in sides]
                + ["—" if sym is None else f"{abs(sym):.1f}%"]
            )
        step = m.overall.symmetry_index_pct.get("step_time")
        if step is not None:
            rows.append([Paragraph("Step time", st["cell"])]
                        + [""] * len(sides) + [f"{abs(step):.1f}%"])
        side_w = content_w * 0.26
        label_w = content_w - side_w * len(sides) - content_w * 0.14
        story.append(_table(rows, [label_w] + [side_w] * len(sides) + [content_w * 0.14]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Values are medians across strides; IQR is the spread between "
            "strides. Asymmetry of 0% is perfectly symmetric. Confidence "
            "(high / medium / low / unvalidated) reflects how well the "
            "measurement is validated, not how good the result is.",
            st["small"]))

    # Charts
    chart = build_comparison_chart(results)
    if chart:
        story.append(KeepTogether([
            Paragraph("Charts", st["h2"]),
            Paragraph("Left vs. right medians", st["body"]),
            Spacer(1, 4),
            _image(chart, content_w, 12 * cm),
        ]))
    cycles = chart_paths.get("cycles_plot")
    if cycles and Path(cycles).exists():
        story.append(Spacer(1, 10))
        story.append(KeepTogether([
            Paragraph("Stride cycles: every valid stride overlaid, contact to "
                      "next contact, with the median in black", st["body"]),
            Spacer(1, 4),
            _image(str(cycles), content_w, 12 * cm),
        ]))
    debug = chart_paths.get("debug_plot")
    if debug and Path(debug).exists():
        story.append(PageBreak())
        story.append(Paragraph("Appendix: detected foot events", st["h2"]))
        story.append(Paragraph(
            "Ankle position relative to the pelvis over time, with the "
            "detected initial contacts and toe-offs. Use it to check the "
            "detection looks right.", st["small"]))
        story.append(Spacer(1, 4))
        story.append(_image(str(debug), content_w, 14 * cm))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, DISCLAIMER)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    out = io.BytesIO()
    SimpleDocTemplate(
        out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title="FormBack running form report", author="FormBack",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()
