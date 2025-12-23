import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# Data
# -----------------------------
index_computation = [6.02, 1.2]
index_compression = [2.99, 0.24]
hessian_computation = [20.63, 38.28]

data = [
    [index_computation[0], index_compression[0], hessian_computation[0]],
    [index_computation[1], index_compression[1], hessian_computation[1]],
]

labels = ["Index Computation", "Index Compression", "Hessian Computation"]

# -----------------------------
# Seaborn style
# -----------------------------
sns.set_theme(style="white", context="paper", font_scale=1.3)
colors = sns.color_palette("Set2", 3)

# -----------------------------
# Pie with aggressive separation
# -----------------------------
def pie_with_leaders(ax, values, title):
    total = sum(values)
    wedges, _ = ax.pie(
        values,
        startangle=90,
        colors=colors,
        wedgeprops=dict(edgecolor="white", linewidth=1)
    )

    label_positions = []

    for wedge, value in zip(wedges, values):
        angle = (wedge.theta1 + wedge.theta2) / 2
        rad = np.deg2rad(angle)

        # Pie edge
        x0, y0 = np.cos(rad), np.sin(rad)

        # Diagonal point (farther out)
        x1, y1 = 1.25 * x0, 1.25 * y0

        # Horizontal end (much farther)
        x2 = 1.75 if x1 > 0 else -1.75
        y2 = y1

        # Enforce vertical separation
        for _, prev_y in label_positions:
            if abs(y2 - prev_y) < 0.14:
                y2 += 0.18 * np.sign(y2 - prev_y if y2 != prev_y else 1)

        label_positions.append((x2, y2))

        percent = 100 * value / total
        text = f"{percent:.3f}%"

        # Leader line
        ax.plot([x0, x1, x2], [y0, y1, y2], lw=1)

        # Label
        ax.text(
            x2,
            y2,
            text,
            ha="left" if x2 > 0 else "right",
            va="center",
            fontsize=12,
            weight="bold"
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_aspect("equal")


# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

pie_with_leaders(axes[0], data[0], "Static")
pie_with_leaders(axes[1], data[1], "Dynamic")

fig.legend(
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=12
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
