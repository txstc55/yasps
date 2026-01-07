import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})
# -----------------------------
# Data
# -----------------------------
data = {
    "Ours": {
        "With Determinant": 0.59,
        "Without Determinant": 0.5594,
    },
    "PyTorch": {
        "With Determinant": 216.93,
        "Without Determinant": 14.597,
    },
    "JAX": {
        "With Determinant": 10.70,
        "Without Determinant": 6.764,
    },
    "SymPy": {
        "With Determinant": 1.027440,
        "Without Determinant": 0.829235,
    },
}

# -----------------------------
# Normalize by Ours (With Determinant)
# -----------------------------
baseline = data["Ours"]["With Determinant"]

conditions = ["Without Determinant", "With Determinant"]   # y-axis categories
backends = ["Ours", "PyTorch", "JAX", "SymPy"][::-1]              # 4 bars per category

# Build a 2x4 array: rows=conditions, cols=backends
vals = np.array([
    [data[b][cond] / baseline for b in backends]
    for cond in conditions
], dtype=float)

# -----------------------------
# Style (SIGGRAPH-ish)
# -----------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
})

fig, ax = plt.subplots(figsize=(3.75, 1.8), dpi=300, constrained_layout=True)

colors = {
  "Ours": "#edae49",
  "PyTorch": "#d1495b",
  "JAX": "#00798c",
  "SymPy": "#6a4c93"
}

# -----------------------------
# Horizontal grouped bars
# -----------------------------
y = np.arange(len(conditions))
group_height = 0.70
bar_h = group_height / len(backends)
offsets = (np.arange(len(backends)) - (len(backends) - 1) / 2.0) * bar_h

for j, b in enumerate(backends):
    xvals = vals[:, j]
    ax.barh(
        y + offsets[j],
        xvals,
        height=bar_h * 0.92,
        color=colors[b],
        edgecolor="black",
        linewidth=0.6,
        label=b,
    )

    # Annotate slowdown values (x)
    for yi, xv in zip(y + offsets[j], xvals):
        ax.text(
            xv * 1.05, yi, f"{xv:.1f}×",
            va="center", ha="left",
            fontsize=6, fontweight="semibold",
        )

# -----------------------------
# Axes: log scale, clean look
# -----------------------------
ax.set_yticks(y)
ax.set_yticklabels(conditions)

ax.set_xscale("log")

# Put a reasonable left bound so labels don't bunch at 0
xmin = max(0.08, np.min(vals) / 1.5)
xmax = np.max(vals) * 1.8
ax.set_xlim(xmin, xmax)

ax.grid(False)
ax.set_xlabel("")   # you can set a label if you want
ax.set_ylabel("")

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)

# Optional: hide x tick labels (uncomment if you want the clean/no-ticks look)
ax.set_xticklabels([])
ax.tick_params(axis="x", which="both", length=0)
# -----------------------------
# Legend (inside, centered)
# -----------------------------
ax.legend(
    loc="center",
    frameon=False,
    ncol=2,
    handlelength=1.1,
    handletextpad=0.4,
    labelspacing=0.25,
    columnspacing=0.8,
)


plt.show()
