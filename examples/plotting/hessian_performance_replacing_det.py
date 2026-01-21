import numpy as np
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
# Baseline: Ours (With Determinant)
# -----------------------------
baseline = data["Ours"]["With Determinant"]

# Backends (order top->bottom in barh; reverse if you want)
backends = ["Ours", "PyTorch", "JAX", "SymPy"][::-1]

# -----------------------------
# "Determinant overhead" (With - Without), normalized by baseline
# This is the single value you want to show as one set of bars.
# -----------------------------
delta = np.array(
    [(data[b]["With Determinant"] - data[b]["Without Determinant"]) / data[b]["With Determinant"] for b in backends],
    dtype=float
)

# -----------------------------
# Style (SIGGRAPH-ish)
# -----------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
})

fig, ax = plt.subplots(figsize=(3.75, 0.8), dpi=300, constrained_layout=True)

colors = {
    "Ours": "#edae49",
    "PyTorch": "#d1495b",
    "JAX": "#00798c",
    "SymPy": "#6a4c93",
}

# -----------------------------
# Single horizontal bar set
# -----------------------------
y = np.arange(len(backends))
bars = ax.barh(
    y,
    delta,
    height=0.85,
    color=[colors[b] for b in backends],
    edgecolor="black",
    linewidth=0.6,
)

import matplotlib.transforms as mtransforms
for yi, xv in zip(y, delta):
    trans = mtransforms.offset_copy(
        ax.transData,
        fig=fig,
        x=4 if xv >= 0 else -4,  # 4 px away from bar end
        units='points'
    )

    ax.text(
        xv, yi - 0.05,
        f"{abs(xv * 100.0):.2f}%",
        transform=trans,
        va="center",
        ha="left" if xv >= 0 else "right",
        fontsize=6,
        fontweight="semibold",
    )

# -----------------------------
# Axes (clean look)
# -----------------------------
ax.set_yticks(y)
ax.set_yticklabels(backends)
ax.set_xlabel("")  # e.g. "Determinant overhead (normalized to Ours w/ det)" if you want
ax.set_ylabel("")
ax.set_xticks([])
ax.set_xscale("log")

# Optional: put 0 line in the middle so negative would be visible (if it happens)
# ax.axvline(0, linewidth=0.8, color="black", alpha=0.6)

# Limits with a little padding
xmin = min(-0.05, delta.min() * 1.15)
xmax = max(0.05, delta.max() * 1.15)
ax.set_xlim(xmin, xmax)

ax.grid(False)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)
ax.set_xticklabels([])  # hide tick labels for the super-clean look
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", length=0)

# Optional: hide x tick labels (uncomment if you want the clean/no-ticks look)
ax.set_xticklabels([])
ax.tick_params(axis="x", which="both", length=0)

plt.show()
