import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})
# -----------------------------
# Input LOC numbers
# -----------------------------
loc = {
    "Soft + Cloth": {
        "STARK": 77,
        "GIPC": 68,
    },
    "Soft + Cloth\n+ Stiff": {
        "STARK": 77 + 45,
        "GIPC": 121,
    },
}

# -----------------------------
# Build dataframe
# rows = methods, cols = cases
# -----------------------------
df = pd.DataFrame(loc)
methods = df.index.tolist()
cases = df.columns.tolist()

vals = df.values.astype(float)

# -----------------------------
# Plot settings (SIGGRAPH-friendly)
# -----------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

fig, ax = plt.subplots(
    figsize=(3.375, 2.2),
    dpi=300,
    constrained_layout=True
)

# Colors for cases (bars)
colors = {
    "Soft + Cloth": "#118ab2",
    "Soft + Cloth\n+ Stiff": "#ef476f",
}

# -----------------------------
# Grouped bars
# -----------------------------
x = np.arange(len(methods))
group_width = 0.75
bar_w = group_width / len(cases)
offsets = (np.arange(len(cases)) - (len(cases) - 1) / 2.0) * bar_w

for j, case in enumerate(cases):
    y = vals[:, j]
    ax.bar(
        x + offsets[j],
        y,
        width=bar_w * 0.95,
        label=case,
        color=colors[case],
        edgecolor="black",
        linewidth=0.6,
    )

    # Value labels
    for xi, yi in zip(x + offsets[j], y):
        ax.text(
            xi, yi + 1.0,
            f"{int(yi)}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="semibold",
        )

# -----------------------------
# Styling
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])
ax.grid(False)

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0)

# Legend INSIDE the plot (top-left, clean)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    frameon=False,
    handlelength=1.2,
    labelspacing=0.4,
    ncol = 2
)

# Tight y-limits
ymax = np.max(vals)
ax.set_ylim(0, ymax * 1.18)

plt.show()
