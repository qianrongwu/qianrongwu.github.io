"""Render the two figures for section 5 from results.json (SVG + PNG)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results.json"))
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"human_only": ("#2a78d6", "Human-only"), "plug_in": ("#eb6834", "Plug-in"),
          "ppi": ("#1baf7a", "PPI"), "ppi_pp": ("#eda100", "PPI++")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ------------------------------------------------------------------ Figure A
A = R["experiment_a"]
x = [100 * r["plug_in_true_bias"] for r in A]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 4.0))
for key, (color, name) in SERIES.items():
    a1.plot(x, [100 * r[key]["coverage"] for r in A], color=color, lw=2, label=name,
            marker="o", ms=3.5, zorder=3 if key == "plug_in" else 2)
    a2.plot(x, [100 * r[key]["ci_width"] for r in A], color=color, lw=2, label=name)
a1.axhline(95, color=MUTED, lw=1, ls=(0, (4, 3)))
a1.text(-12, 96.5, "nominal 95%", color=MUTED, fontsize=9, va="bottom")
a1.annotate("plug-in covers only when\nFP and FN errors cancel", xy=(0.6, 60), xytext=(3.2, 52),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=AXIS))
a1.set_ylim(-3, 106)
a1.set_title("Coverage of nominal 95% intervals", loc="left", fontsize=11, color=INK)
a1.set_ylabel("coverage (%)")
a2.set_title("Mean interval width", loc="left", fontsize=11, color=INK)
a2.set_ylabel("width (percentage points)")
a2.set_ylim(0, 10)
for key in SERIES:
    yv = 100 * A[-1][key]["ci_width"]
    a2.text(12.4, yv, SERIES[key][1], color=INK2, fontsize=9, va="center")
for ax in (a1, a2):
    style(ax)
    ax.set_xlabel("net label bias of the model (pp)")
    ax.set_xlim(-12.8, 12.8)
a2.set_xlim(-12.8, 17)
for ax in (a1, a2):
    ax.set_xticks([-10, -5, 0, 5, 10])
a1.legend(frameon=False, loc="center left", fontsize=9, labelcolor=INK2)
fig.suptitle("Model accuracy is fixed at 88%; only the split of its errors changes "
             "(net bias = false-positive mass − false-negative mass)",
             x=0.012, ha="left", fontsize=9.5, color=INK2)
fig.tight_layout(w_pad=3, rect=(0, 0, 1, 0.95))
for ext in ("svg", "png"):
    fig.savefig(f"fig_coverage_vs_bias.{ext}", dpi=200, facecolor=SURFACE)

# ------------------------------------------------------------------ Figure B
B = R["experiment_b"]
rows = [("routed_human_only", "Human-only, routed sample\n(naive)"),
        ("routed_naive_ppi", "PPI, routed sample\n(treated as random)"),
        ("routed_ipw", "IPW-corrected, routed sample\n(known sampling prob.)"),
        ("uniform_ipw", "Uniform sample, same budget\n(reference)")]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
specs = [("coverage", "Coverage (%)", 100, (0, 108), "{:.1f}"),
         ("bias", "Bias (pp)", 100, (-7, 11.5), "{:+.1f}"),
         ("ci_width", "Mean interval width (pp)", 100, (0, 14), "{:.1f}")]
ypos = list(range(len(rows)))[::-1]
for ax, (metric, title, mult, lim, fmt) in zip(axes, specs):
    vals = [mult * B[k][metric] for k, _ in rows]
    ax.barh(ypos, vals, height=0.5, color="#2a78d6")
    for yv, v in zip(ypos, vals):
        off = 0.02 * (lim[1] - lim[0])
        lab = "0.0" if abs(v) < 0.05 else fmt.format(v)
        ax.text(v + off if v >= -0.05 else v - off, yv, lab, va="center",
                ha="left" if v >= -0.05 else "right", fontsize=9, color=INK)
    ax.set_xlim(*lim)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if metric == "coverage":
        ax.axvline(95, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(95, 3.42, "nominal 95% ", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    if metric == "bias":
        ax.axvline(0, color=AXIS, lw=1)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([lab for _, lab in rows], color=INK2, fontsize=9)
axes[0].set_ylim(-0.5, 3.75)
fig.tight_layout(w_pad=2)
for ext in ("svg", "png"):
    fig.savefig(f"fig_routed_sampling.{ext}", dpi=200, facecolor=SURFACE)
print("ok")
