"""Altair charts for the Streamlit app.

Colours come from the same validated palette as the report figures, with separate steps
for light and dark mode. Every chart has tooltips, and line charts use a nearest-point
hover rule so values are readable without landing exactly on a line.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]  # categorical slots in fixed order
    increase: str  # SHAP: pushes towards churn
    decrease: str  # SHAP: pushes away from churn
    context: str  # baselines, de-emphasised series
    ink: str  # reference rules and their labels


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    increase="#e34948",
    decrease="#2a78d6",
    context="#b5b4ad",
    ink="#52514e",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    increase="#e66767",
    decrease="#3987e5",
    context="#6f6e69",
    ink="#c3c2b7",
)


# Tenure axis shared by the survival charts: 0-72 months, ticks every year.
TENURE_X = alt.X(
    "month:Q",
    title="Tenure (months)",
    scale=alt.Scale(domain=[0, 72], nice=False),
    axis=alt.Axis(values=list(range(0, 73, 12))),
)


def palette(theme_type: str | None) -> Palette:
    return DARK if theme_type == "dark" else LIGHT


def _hover_rule(data: pd.DataFrame, x: str, x_title: str, tooltip: list, color: str):
    """Vertical crosshair that snaps to the nearest x and shows a tooltip."""
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False)
    rule = (
        alt.Chart(data)
        .mark_rule(color=color, strokeWidth=1)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            opacity=alt.condition(nearest, alt.value(0.6), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(nearest)
    )
    return rule


def survival_curve(curve: pd.DataFrame, pal: Palette, baseline: pd.DataFrame | None = None):
    """Conditional survival from today's tenure; optional pinned baseline in gray."""
    frames = [curve.assign(scenario="Current")]
    colors = [pal.series[0]]
    if baseline is not None:
        frames.append(baseline.assign(scenario="Pinned baseline"))
        colors.append(pal.context)
    data = pd.concat(frames, ignore_index=True)
    domain = list(dict.fromkeys(data["scenario"]))

    lines = (
        alt.Chart(data)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=TENURE_X,
            y=alt.Y(
                "survival:Q",
                title="P(still a customer | active today)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "scenario:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(title=None, orient="bottom") if len(domain) > 1 else None,
            ),
        )
    )
    tooltip = [
        alt.Tooltip("month:Q", title="Month"),
        alt.Tooltip("survival:Q", title="Still a customer", format=".0%"),
    ]
    hover = _hover_rule(curve, "month", "Tenure (months)", tooltip, pal.ink)
    return (lines + hover).properties(height=300)


def shap_drivers(drivers: pd.DataFrame, pal: Palette):
    """Diverging bars of SHAP contributions (log-odds), largest first."""
    value = drivers["value"].map(_format_value)  # mixed str/float would break Arrow
    data = drivers.assign(
        value=value,
        label=drivers["feature"] + " = " + value,
        effect=drivers["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk"),
    )
    order = data.sort_values("contribution", key=abs, ascending=False)["label"].tolist()
    bars = (
        alt.Chart(data)
        .mark_bar(size=16, cornerRadiusEnd=4)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("contribution:Q", title="SHAP contribution (log-odds of churn)"),
            color=alt.Color(
                "effect:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=[pal.increase, pal.decrease]
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Attribute"),
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=pal.context).encode(x="x:Q")
    return (bars + zero).properties(height=34 * len(data) + 40)


def profit_curve(curve: pd.DataFrame, threshold: float, pal: Palette):
    """Expected campaign profit vs targeting threshold, with the selected threshold marked."""
    tooltip = [
        alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
        alt.Tooltip("n_targeted:Q", title="Customers targeted", format=","),
        alt.Tooltip("churners_reached:Q", title="Churners reached", format=",.0f"),
        alt.Tooltip("profit:Q", title="Expected profit", format="$,.0f"),
    ]
    line = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2, color=pal.series[0])
        .encode(
            x=alt.X("threshold:Q", title="Target customers with P(churn) at or above"),
            y=alt.Y("profit:Q", title="Expected profit (USD)", axis=alt.Axis(format="$,.0f")),
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=pal.context).encode(y="y:Q")
    chosen = alt.Chart(pd.DataFrame({"threshold": [threshold]}))
    rule = chosen.mark_rule(color=pal.ink, strokeWidth=1).encode(x="threshold:Q")
    label = chosen.mark_text(align="left", dx=4, dy=-6, color=pal.ink, fontSize=11).encode(
        x="threshold:Q", y=alt.value(12), text=alt.value(f"selected {threshold:.2f}")
    )
    hover = _hover_rule(
        curve, "threshold", "Target customers with P(churn) at or above", tooltip, pal.ink
    )
    return (zero + line + rule + label + hover).properties(height=320)


def segment_survival(curves: pd.DataFrame, segment: str, pal: Palette):
    """Survival curves by segment level (long frame: level, month, survival)."""
    levels = list(dict.fromkeys(curves["level"]))
    lines = (
        alt.Chart(curves)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=TENURE_X,
            y=alt.Y(
                "survival:Q",
                title="Share still active",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "level:N",
                title=segment,
                scale=alt.Scale(domain=levels, range=list(pal.series[: len(levels)])),
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("level:N", title=segment),
                alt.Tooltip("month:Q", title="Month"),
                alt.Tooltip("survival:Q", title="Still active", format=".1%"),
            ],
        )
    )
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["month"], empty=False)
    points = (
        alt.Chart(curves)
        .mark_point(size=60, filled=True)
        .encode(
            x="month:Q",
            y="survival:Q",
            color=alt.Color(
                "level:N", scale=alt.Scale(domain=levels, range=list(pal.series)), legend=None
            ),
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("level:N", title=segment),
                alt.Tooltip("month:Q", title="Month"),
                alt.Tooltip("survival:Q", title="Still active", format=".1%"),
            ],
        )
        .add_params(nearest)
    )
    return (lines + points).properties(height=360)


def _format_value(v) -> str:
    if isinstance(v, str):
        return v
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"
