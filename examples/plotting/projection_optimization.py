import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (ms)
# -----------------------------
energies = ["Baraff-Witkin", "Stable Neo-Hookean"]
variants = ["Original", "Modified"]

# (no_proj, with_proj)
data = {
    "Baraff-Witkin": {
        "Original": (0.11, 0.54),
        "Modified": (0.13, 0.27),
    },
    "Stable Neo-Hookean": {
        "Original": (0.20, 1.106),
        "Modified": (0.30, 0.71),
    },
}

# Flatten into arrays aligned with (energy, variant)
base = []
evd = []
total = []

for e in energies:
    for v in variants:
        no_p, with_p = data[e][v]
        base.append(no_p)
        evd_over = max(0.0, with_p - no_p)  # guard
        evd.append(evd_over)
        total.append(with_p)

base = np.array(base, dtype=float)
evd = np.array(evd, dtype=float)
total = np.array(total, dtype=float)

# Helper to index into flattened arrays
def idx(energy_i, variant_i):
    return energy_i * len(variants) + variant_i

# -----------------------------
# Plot style (SIGGRAPH-ish)
# -----------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
})

fig, ax = plt.subplots(figsize=(3.75, 2.2), dpi=300, constrained_layout=True)

# Colors
c_base = "#118ab2"   # base (no projection)
c_evd  = "#ef476f"   # EVD overhead

# -----------------------------
# Layout: two groups (energies), each with two bars (variants)
# -----------------------------
x = np.arange(len(energies))
bar_w = 0.32
offsets = np.array([-0.18, +0.18])  # Original, Modified

# Draw stacked bars
for vi, v in enumerate(variants):
    xs = x + offsets[vi]
    b = np.array([base[idx(ei, vi)] for ei in range(len(energies))])
    o = np.array([evd[idx(ei, vi)]  for ei in range(len(energies))])
    t = b + o

    # Bars
    ax.bar(
        xs, b, width=bar_w,
        color=c_base, edgecolor="black", linewidth=0.6,
        label="Hessian (no projection)" if vi == 0 else None
    )
    ax.bar(
        xs, o, bottom=b, width=bar_w,
        color=c_evd, edgecolor="black", linewidth=0.6,
        label="EVD (projection overhead)" if vi == 0 else None
    )

    # Percent labels INSIDE each segment + total time on TOP
    for xi, bb, oo, tt in zip(xs, b, o, t):
        if tt <= 0:
            continue

        pct_base = 100.0 * bb / tt
        pct_evd  = 100.0 * oo / tt

        # Base percent (centered inside base segment)
        if bb > 0:
            ax.text(
                xi, bb * 0.5,
                f"{pct_base:.0f}%",
                ha="center", va="center",
                fontsize=6, fontweight="semibold"
            )

        # EVD percent (centered inside EVD segment)
        if oo > 0:
            ax.text(
                xi, bb + oo * 0.5,
                f"{pct_evd:.0f}%",
                ha="center", va="center",
                fontsize=6, fontweight="semibold"
            )

        # Total time on top
        ax.text(
            xi, tt * 1.03,
            f"{tt:.3g} ms",
            ha="center", va="bottom",
            fontsize=6, fontweight="semibold"
        )

# -----------------------------
# Axes styling
# -----------------------------
ax.set_xticks(x)
ax.set_xticklabels([])   # hide energy names on axis
ax.set_ylabel("")
ax.set_yticks([])
ax.grid(False)

for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0)

# Legend inside
ax.legend(
    loc="center left",
    frameon=False,
    ncol=1,
    handlelength=1.1,
    handletextpad=0.4,
    labelspacing=0.25,
    bbox_to_anchor=(0.02, 0.72)
)

# Variant labels under each bar
neg_y = -0.03 * (base.max() + evd.max() + 1e-9)
for ei, e in enumerate(energies):
    for vi, v in enumerate(variants):
        xi = x[ei] + offsets[vi]
        ax.text(
            xi, neg_y,
            v,
            ha="center", va="top",
            fontsize=6
        )

# Y-limit with headroom for top labels
ymax = (base + evd).max()
ax.set_ylim(0, ymax * 1.25)

# Energy names on top of each group
for xi, energy in zip(x, energies):
    ax.text(
        xi,
        ymax * 1.14,
        energy,
        ha="center", va="bottom",
        fontsize=6,
    )

# plt.show()
plt.savefig(
  "projection_optimization.pdf",
  format="pdf",
  bbox_inches="tight",
  pad_inches=0.00
)
plt.close()
