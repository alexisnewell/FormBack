"""
Insights engine: turns measured metrics into plain-language observations.

This is a port of ``buildInsights`` in frontend/src/app.tsx, so the PDF says
the same things the Insights tab does. If you change a threshold or a sentence
in one place, change it in the other (or move the web UI onto this module).

Rounding follows JavaScript: ``toFixed`` and ``Math.round`` round halves up,
Python's built-ins round halves to even, so the helpers below imitate JS.
"""

import math
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Dict, List, Optional

from api.schemas.analysis import Diagnostics, MetricStat, Metrics

CAUTION = (
    " This metric has lower measurement confidence, so treat it as a rough guide."
)
TONE_ORDER = {"warn": 0, "info": 1, "good": 2}


@dataclass
class Insight:
    id: str
    tone: str  # "good" | "info" | "warn"
    title: str
    detail: str
    facts: List[str] = field(default_factory=list)


def _fixed(value: float, digits: int) -> str:
    """JavaScript ``Number.prototype.toFixed``."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(value).quantize(q, rounding=ROUND_HALF_UP))


def _round(value: float) -> int:
    """JavaScript ``Math.round``."""
    return math.floor(value + 0.5)


def _prettify(key: str) -> str:
    s = key.replace("_", " ")
    return s[:1].upper() + s[1:]


def build_insights(m: Metrics, diag: Optional[Diagnostics]) -> List[Insight]:
    out: List[Insight] = []
    sides = list(m.per_side)

    def stats(k: str) -> List[MetricStat]:
        return [m.per_side[s][k] for s in sides if k in m.per_side[s]]

    def mean(k: str) -> Optional[float]:
        v = stats(k)
        return sum(x.median for x in v) / len(v) if v else None

    def shaky(k: str) -> bool:
        v = stats(k)
        return bool(v) and all(
            x.confidence is not None and x.confidence.value in ("low", "unvalidated")
            for x in v
        )

    def label(k: str) -> str:
        if k == "step_time":
            return "Step time"
        v = stats(k)
        return v[0].label if v else _prettify(k)

    def add(i: Insight, based_on: Optional[str] = None) -> None:
        """Soften a "warn" to "info" and add a caution when the metric is shaky."""
        if based_on and shaky(based_on):
            i.tone = "info" if i.tone == "warn" else i.tone
            i.detail += CAUTION
        out.append(i)

    # Data sufficiency
    stride_counts = (diag.stride_counts if diag else {}) or {}
    counts = list(stride_counts.values())
    if counts and min(counts) < 10:
        out.append(Insight(
            "few-strides", "warn", "Short sample",
            "Fewer than 10 strides were measured on at least one leg, so the "
            "numbers can swing a lot. A longer clip (20+ strides per leg) "
            "gives a steadier picture.",
            [f"{_prettify(s)}: {stride_counts.get(s, 0)} strides" for s in sides],
        ))

    # Cadence
    cadence = m.overall.cadence_spm
    if cadence is not None:
        c = _round(cadence)
        if cadence < 150:
            add(Insight(
                "cadence", "info", "Cadence is on the lower side",
                "Recreational runners average about 165 steps per minute at "
                "easy paces, and cadence rises with speed, so a low number at "
                "a slow pace is common. A modest increase (about 5%) shortens "
                "each stride and often reduces braking.",
                [f"{c} spm"]))
        elif cadence <= 185:
            add(Insight(
                "cadence", "good", "Cadence is in a typical range",
                "Most recreational runners land between about 155 and 175 at "
                "easy paces, higher when faster. Compare between clips at a "
                "similar pace.",
                [f"{c} spm"]))
        else:
            add(Insight(
                "cadence", "info", "Cadence is high",
                "A high step rate is common at fast paces and for shorter "
                "runners. If this was an easy run, check that the detected "
                "steps are correct.",
                [f"{c} spm"]))

    # Symmetry
    sym = sorted(
        ((k, abs(v)) for k, v in m.overall.symmetry_index_pct.items() if v is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if sym:
        big = [x for x in sym if x[1] >= 10]
        mild = [x for x in sym if 5 <= x[1] < 10]

        def sym_facts(items) -> List[str]:
            return [f"{label(k)}: {_fixed(v, 1)}%" for k, v in items[:4]]

        if big:
            what = label(big[0][0]).lower() if len(big) == 1 else f"{len(big)} metrics"
            add(
                Insight(
                    "symmetry", "warn", f"Left and right differ in {what}",
                    "Some difference between legs is normal, and in a large "
                    "study of recreational runners asymmetry wasn't linked to "
                    "injury risk. Video measurement adds its own error, so "
                    "film a second clip before reading much into it.",
                    sym_facts(big)),
                big[0][0] if big[0][0] != "step_time" else None,
            )
        elif mild:
            add(Insight(
                "symmetry", "info", "Mild left/right differences",
                "A little asymmetry is normal. Nothing here is large enough "
                "to be a concern on its own.",
                sym_facts(mild)))
        else:
            out.append(Insight(
                "symmetry", "good", "Left and right look well balanced",
                "Every measured metric differs by less than 5% between legs.",
                [f"Largest: {label(sym[0][0])} {_fixed(sym[0][1], 1)}%"]))

    # Foot strike + overstride
    pattern = m.overall.foot_strike_pattern
    angle = mean("foot_strike_angle_deg")
    ahead = mean("ankle_ahead_of_pelvis_leg_lengths")
    if pattern:
        text: Dict[str, str] = {
            "rearfoot": (
                "Your heel tends to land first. This is the most common "
                "pattern and isn\u2019t a problem by itself. What matters more "
                "is how far in front of your body the foot lands."),
            "midfoot": (
                "Your foot tends to land flat, which spreads impact across "
                "the foot."),
            "forefoot": (
                "You tend to land on the ball of the foot. This loads the "
                "calf and Achilles more, so build up gradually if you\u2019re "
                "changing toward it."),
        }
        add(
            Insight(
                "foot-strike", "info", f"{_prettify(pattern)} strike",
                text.get(pattern, ""),
                [f"Foot angle at contact: {_fixed(angle, 1)}\u00b0"]
                if angle is not None else []),
            "foot_strike_angle_deg",
        )
    else:
        out.append(Insight(
            "foot-strike", "info", "Foot strike varies from stride to stride",
            "Your landing pattern wasn\u2019t consistent enough to label. This "
            "is common and can also reflect video angle or frame rate."))

    if ahead is not None:
        add(
            Insight(
                "overstride", "info", "Foot placement relative to your hips",
                "This is how far the ankle is ahead of the pelvis at contact, "
                "in leg lengths. Higher values mean the foot lands further in "
                "front of you. There's no agreed cutoff for overstriding, so "
                "track it across clips rather than judging a single number.",
                [f"{_fixed(ahead, 2)} leg lengths ahead"]),
            "ankle_ahead_of_pelvis_leg_lengths",
        )

    # Trunk lean
    lean = mean("mean_trunk_lean_deg")
    if lean is not None:
        l = abs(lean)
        fact = [f"{_fixed(lean, 1)}\u00b0 trunk lean"]
        if l < 2:
            add(Insight(
                "lean", "info", "Very upright posture",
                "A larger lean shifts load from the knee toward the hip and "
                "can raise energy cost. Check whether it comes from the hips "
                "or from the whole body tilting, and whether it eases at a "
                "slower pace.", fact), "mean_trunk_lean_deg")
        elif l <= 12:
            add(Insight(
                "lean", "good", "Natural forward lean",
                "A small forward lean like this is typical and generally "
                "efficient.", fact), "mean_trunk_lean_deg")
        else:
            add(Insight(
                "lean", "warn", "Pronounced forward lean",
                "Leaning this far can load the lower back and hip flexors. "
                "Check whether it\u2019s from the hips (bending at the waist) or "
                "from the whole body tilting, and whether it eases at a "
                "slower pace.", fact), "mean_trunk_lean_deg")

    # Vertical oscillation
    osc = mean("vertical_oscillation_pct_leg")
    if osc is not None:
        fact = [f"{_fixed(osc, 1)}% of leg length"]
        if osc <= 10:
            add(Insight(
                "bounce", "good", "Smooth vertical motion",
                "Your body moves up and down relatively little, so more of "
                "your effort goes forward.", fact), "vertical_oscillation_pct_leg")
        elif osc <= 13:
            add(Insight(
                "bounce", "info", "Moderate vertical bounce",
                "A bit of bounce is normal, and it increases with speed. "
                "Staying low and quick off the ground can trim it.", fact),
                "vertical_oscillation_pct_leg")
        else:
            add(Insight(
                "bounce", "warn", "High vertical bounce",
                "You\u2019re spending noticeable energy moving up rather than "
                "forward. Higher cadence and a lighter, quicker push-off "
                "usually reduce this.", fact), "vertical_oscillation_pct_leg")

    # Ground contact time
    gct = mean("contact_time_s")
    if gct is not None:
        fact = [f"{_round(gct * 1000)} ms"]
        if gct > 0.3:
            add(Insight(
                "contact", "info", "Longer ground contact",
                "Contact times above about 300 ms are common at easy paces. "
                "Faster running and a higher cadence tend to shorten them.",
                fact), "contact_time_s")
        else:
            add(Insight(
                "contact", "good", "Quick ground contact",
                "Your feet leave the ground promptly, which is a good sign "
                "for efficiency.", fact), "contact_time_s")

    return sorted(out, key=lambda i: TONE_ORDER[i.tone])  # stable, like JS
