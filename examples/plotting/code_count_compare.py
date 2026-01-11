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
        "YASPS": 776
    },
    "Soft + Cloth\n+ Stiff": {
        "STARK": 77 + 45,
        "GIPC": 121,
        "YASPS": 900
    },
}

df = pd.DataFrame(loc)

# -----------------------------
# Compute incremental LOC
# -----------------------------
delta = df["Soft + Cloth\n+ Stiff"] - df["Soft + Cloth"]
delta = delta.astype(float)

methods = delta.index.tolist()
vals = delta.values

# -----------------------------
# Plot settings (SIGGRAPH-friendly)
# -----------------------------
plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

fig, ax = plt.subplots(
    figsize=(3.375, 1.0),
    dpi=300,
    constrained_layout=True
)

# Colors per method (optional but helps reading)
colors = {
    "STARK": "#8ecae6",
    "GIPC": "#ffb703",
    "YASPS": "#ef476f",
}

# y = np.arange(len(methods))
y_spacing = 0.75          # <--- smaller = bars closer together
bar_height = 0.65         # <--- larger = thicker bars

y = np.arange(len(methods)) * y_spacing
ax.barh(
    y,
    vals,
    height=bar_height,
    color=[colors[m] for m in methods],
    edgecolor="black",
    linewidth=0.6,
)

# -----------------------------
# Value labels
# -----------------------------
for yi, vi in zip(y, vals):
    ax.text(
        vi + max(vals) * 0.02,
        yi,
        f"+{int(vi)}",
        va="center",
        ha="left",
        fontsize=7,
        fontweight="semibold",
    )

# -----------------------------
# Styling
# -----------------------------
ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])

ax.grid(False)
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0)

# Tight x-limits
ax.set_xlim(0, max(vals) * 1.25)

plt.show()
