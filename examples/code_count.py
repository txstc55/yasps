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
parts = [
    "Setup",
    "YASPS Integration",
    "Energy Definition",
    "Collision Detection Initialization",
    "Visualization",
    "Newton Iteration",
]

simulation_names = [
    "Soft + Cloth",
    "Soft + Cloth\n+ ABD",
    "Soft + Cloth\n+ ABD\n+ Cage",
]

data = {
    "Setup": [197 + 59 + 34, 95 + 229, 95 + 277],
    "YASPS Integration": [124, 181, 231],
    "Energy Definition": [38 + 161 + 16, 48 + 160 + 18, 77 + 160],
    "Collision Detection Initialization": [33, 41, 48],
    "Visualization": [24, 27, 32],
    "Newton Iteration": [90, 101, 118],
}

df = pd.DataFrame(data, index=simulation_names)

custom_colors = {
    "Setup": "#ef476f",
    "YASPS Integration": "#f78c6b",
    "Energy Definition": "#ffd166",
    "Collision Detection Initialization": "#06d6a0",
    "Visualization": "#118ab2",
    "Newton Iteration": "#073b4c",
}

# -----------------------------
# Horizontal STACKED bar chart + value + percentage per segment
# (Percentage is relative to the total LOC of that simulation.)
# -----------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
})

fig, ax = plt.subplots(figsize=(3.375, 2), dpi=300)

y = np.arange(len(simulation_names))
totals = df[parts].sum(axis=1).values

left = np.zeros(len(simulation_names), dtype=float)

# tweak: if a segment is too small, put its label just outside the segment
min_segment_fraction_for_inside_label = 0.08  # 8% of total
skip_label_stages = {
    "Collision Detection Initialization",
    "Visualization",
    "Newton Iteration"
}
for stage in parts:
    vals = df[stage].values.astype(float)
    bars = ax.barh(
        y,
        vals,
        left=left,
        color=custom_colors[stage],
        edgecolor="black",
        linewidth=0.6,
        label=stage,
    )

    # annotate each segment with "N (P%)"
    for i, (b, v) in enumerate(zip(bars, vals)):
        if v <= 0:
            continue
        if stage in skip_label_stages:
                    continue
        pct = 100.0 * v / totals[i]
        label = f"{int(v)} ({pct:.0f}%)"
        label = f"{pct:.0f}%"
        seg_frac = v / totals[i]
        x0 = left[i]
        x_center = x0 + v / 2.0
        y_center = b.get_y() + b.get_height() / 2.0

        if seg_frac >= min_segment_fraction_for_inside_label:
            ax.text(
                x_center, y_center, label,
                ha="center", va="center",
                fontsize=6, fontweight="semibold",
            )
        else:
            # place just outside to the right of the segment
            ax.text(
                x0 + v + 1.5, y_center, label,
                ha="left", va="center",
                fontsize=6, fontweight="semibold",
            )

    left += vals
total_offset = 16.0  # horizontal offset to the right of the bar

for i, total in enumerate(totals):
    ax.text(
        total + total_offset,
        y[i],
        f"{int(total)} LOC",
        va="center",
        ha="left",
        fontsize=6,
        fontweight="bold",
    )

# Styling (SIGGRAPH-friendly)
ax.set_yticks(y)
ax.set_yticklabels(simulation_names)
ax.invert_yaxis()               # top-to-bottom order like your list
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])               # hide x ticks like your original
ax.grid(False)
ax.tick_params(axis="y", length=0)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)



ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),   # centered above the axes
    ncol=2,
    frameon=False,
    handlelength=1.0,
    columnspacing=0.8,
    labelspacing=0.4,
    fontsize=6,
)

plt.tight_layout()
plt.show()
