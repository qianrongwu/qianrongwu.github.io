"""Render the article's figures from the verified, rounded simulation results.

Requirements: Python 3 and matplotlib (``python -m pip install matplotlib``).
Run: ``python render_figures.py``. All four outputs are written beside this file.

Data source: https://github.com/qianrongwu/labels-are-estimators
These are synthetic results, not measurements of a deployed detector or an
experiment with real human/LLM annotators. The script renders the reported
results; it does not rerun the simulator or recompute its bootstrap interval.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "detectors-inherit-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


OUTPUT_DIR = Path(__file__).resolve().parent
INK = "#22353D"
MUTED = "#62747B"
TEAL = "#187C78"
TERRACOTTA = "#BD6049"
OCHRE = "#AB813D"
GRID = "#E5EAEB"
PAPER = "#FFFFFF"

# Reported percentages, rounded to one decimal place.
RECALL = (
    ("Production labels", 78.0, TERRACOTTA),
    ("Audit, unweighted", 45.4, OCHRE),
    ("Audit, weighted", 56.4, TEAL),
)
TRUE_RECALL = 57.2
WEIGHTED_INTERVAL = (52.1, 60.4)
QUIET_RECALL = (("Score hidden", 35.4, TEAL), ("Score exposed", 20.8, TERRACOTTA))
INCUMBENT_AGREEMENT = (
    ("Score hidden", 84.3, TEAL),
    ("Score exposed", 88.7, TERRACOTTA),
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        # Outlined glyphs keep the intended appearance on readers' devices.
        "svg.fonttype": "path",
        "svg.hashsalt": "detectors-inherit-figures-v1",
    }
)


def prepare_axis(ax, top):
    """Use a shared full percentage scale with quiet, uncluttered grid lines."""
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, top)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis="x", length=0, pad=10, labelsize=12)
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_rows(ax, rows, *, label_size=15, interval=None):
    for index, (label, value, color) in enumerate(rows):
        y = len(rows) - 1 - index
        ax.barh(y, value, height=0.24, color=color, zorder=3)
        ax.text(0, y + 0.24, label, ha="left", va="bottom", fontsize=label_size)
        value_x = value + 2.2
        if interval and index == len(rows) - 1:
            low, high = interval
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt="o",
                markersize=5,
                markerfacecolor=PAPER,
                markeredgecolor=INK,
                markeredgewidth=1.4,
                color=INK,
                capsize=6,
                capthick=1.5,
                elinewidth=1.8,
                zorder=5,
            )
            value_x = high + 2.2
        ax.text(
            value_x,
            y,
            f"{value:.1f}%",
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=INK,
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1.5},
            zorder=6,
        )


def save_figure(fig, stem, title, description):
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=240,
        metadata={"Title": title, "Description": description},
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.svg",
        metadata={
            "Title": title,
            "Description": description,
            "Creator": "render_figures.py / Matplotlib",
            "Date": None,
        },
    )
    plt.close(fig)


def render_recall():
    title = "One model, three recall estimates"
    fig = plt.figure(figsize=(7.2, 5.1))
    fig.text(0.085, 0.925, title, fontsize=19, fontweight="bold", ha="left")
    fig.text(
        0.085, 0.865, "Synthetic example · ground truth is known", fontsize=12, color=MUTED
    )
    ax = fig.add_axes((0.085, 0.19, 0.86, 0.605))
    prepare_axis(ax, top=2.7)
    ax.axvline(TRUE_RECALL, color=INK, linestyle=(0, (4, 3)), linewidth=1.25, zorder=2)
    ax.text(
        TRUE_RECALL,
        2.58,
        "True recall: 57.2%",
        ha="center",
        va="center",
        fontsize=12,
        color=INK,
        bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 3},
        zorder=6,
    )
    draw_rows(ax, RECALL, interval=WEIGHTED_INTERVAL)
    fig.text(
        0.085,
        0.055,
        "Weighted estimate: 56.4%; approx. 95% interval: 52.1–60.4%.",
        fontsize=10.5,
        color=MUTED,
    )
    save_figure(
        fig,
        "figure",
        title,
        "Synthetic detector recall: production labels 78.0%; unweighted audit 45.4%; "
        "weighted audit 56.4% (approximate 95% bootstrap interval 52.1–60.4%); "
        "true recall 57.2%. All values use a 0–100% scale.",
    )


def render_labelers():
    title = "When the labeler sees the score"
    fig = plt.figure(figsize=(7.2, 7.2))
    fig.text(0.085, 0.943, title, fontsize=19, fontweight="bold", ha="left")
    fig.text(
        0.085,
        0.898,
        "Synthetic labeler experiment · weighted audit estimates",
        fontsize=11.5,
        color=MUTED,
    )
    panels = (
        ((0.085, 0.56, 0.86, 0.255), "Quiet-attack recall", QUIET_RECALL),
        ((0.085, 0.16, 0.86, 0.255), "Agreement with incumbent", INCUMBENT_AGREEMENT),
    )
    for rectangle, panel_title, rows in panels:
        ax = fig.add_axes(rectangle)
        prepare_axis(ax, top=1.65)
        ax.set_title(panel_title, loc="left", fontsize=15, fontweight="bold", pad=15)
        draw_rows(ax, rows, label_size=13)
    fig.text(
        0.085,
        0.047,
        "Within the top 10% pool · 20% positive-label budget",
        fontsize=10.5,
        color=MUTED,
    )
    save_figure(
        fig,
        "figure_labelers",
        title,
        "Synthetic labeler results, weighted audit estimates within the top 10% pool "
        "at a 20% positive-label budget. Quiet-attack recall is 35.4% with the incumbent "
        "score hidden and 20.8% with it exposed; agreement with the incumbent is 84.3% "
        "with the score hidden and 88.7% with it exposed. All axes run from 0 to 100%.",
    )


if __name__ == "__main__":
    render_recall()
    render_labelers()
    print("Wrote figure.png, figure.svg, figure_labelers.png, and figure_labelers.svg")
