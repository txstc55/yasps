import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})
labels = [
    "No Optimization (24×24)",
    "Block Compression (15×15)",
    "Fully Optimized (6x6)",
    "Hessian Computation"
]
time = np.array([148.57, 35.99, 3.39, 0.08])

# Normalize by Hessian computation
baseline = time[-1]
ratio = time / baseline

# Paper-friendly colors
colors = ["#8c1d18", "#d95f02", "#1b9e77", "#4d4d4d"]

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6,
    "legend.fontsize": 6.5,
})

sns.set_theme(style="white")

fig, ax = plt.subplots(figsize=(3.75, 0.8), dpi=300, constrained_layout=True)
sns.barplot(
    x=ratio,
    y=labels,
    orient="h",
    palette=colors,
    width=1.0,              # <- no gap between bars
    edgecolor="black",      # <- black border
    linewidth=0.6,
    ax=ax
)

# Log scale
ax.set_xscale("log")

# Bar labels (log-safe offset)
for i, v in enumerate(ratio):
    ax.text(
        v * 1.12,
        i,
        f"{v:.1f}×",
        va="center",
        fontsize=6,
        weight="bold"
    )
ax.tick_params(axis="y", labelsize=6)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(0.9, ratio.max() * 1.6)
ax.set_xticklabels([])

sns.despine(left=True, bottom=True)
fig.tight_layout(pad=0.3)
plt.show()
