import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})
# -----------------------------
# Data
# -----------------------------
methods = ["Optimized", "No Parallel", "Monolithic"]
compute = np.array([32.85, 79.96, 309.37])
hessian = np.array([105.52, 369.78, 872.49])

x = np.arange(len(methods))
width = 0.35

# -----------------------------
# Style
# -----------------------------
sns.set_theme(style="white")
plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(3.75, 2.0), dpi=300)

colors = sns.color_palette("Set2", 2)

bars_compute = ax.bar(
    x - width/2,
    compute,
    width,
    label="Compute",
    color=colors[0],
    edgecolor="black",
    linewidth=0.5
)

bars_hessian = ax.bar(
    x + width/2,
    hessian,
    width,
    label="Hessian",
    color=colors[1],
    edgecolor="black",
    linewidth=0.5
)

# -----------------------------
# Per-bar labels
# -----------------------------
for bar in list(bars_compute) + list(bars_hessian):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.03,
        f"{height:.0f}",
        ha="center",
        va="bottom",
        fontsize=6
    )

# -----------------------------
# Remove axes
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(methods)

ax.tick_params(left=False, bottom=False, labelleft=False)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.legend(
  frameon=False,
  loc="upper left",
  fontsize=6,
  bbox_to_anchor=(0.1, 0.8)
)

plt.tight_layout(pad=0.1)
# plt.show()
plt.savefig(
  "compile_performance.pdf",
  format="pdf",
  bbox_inches="tight",
  pad_inches=0.00
)
plt.close()
