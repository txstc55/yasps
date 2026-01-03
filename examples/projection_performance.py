import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

sns.set_theme(style="white", font_scale=1.2)

fig, ax = plt.subplots(figsize=(7.5, 3.6))
sns.barplot(
    x=ratio,
    y=labels,
    orient="h",
    palette=colors,
    width=1.0,              # <- no gap between bars
    edgecolor="black",      # <- black border
    linewidth=1.2,
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
        fontsize=12,
        weight="bold"
    )

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(0.9, ratio.max() * 1.6)
ax.set_xticklabels([])

sns.despine(left=True, bottom=True)
fig.tight_layout(pad=0.3)
plt.show()
