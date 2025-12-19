import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Data (TOTAL time in ms)
# -----------------------------
data = [
    # Energy, Instances, Ours_G, Ours_H, Torch_G, Torch_H, JAX_G, JAX_H, sympy_g, sympy_h
    ("Bending", 104065, 0.17,     2.81,    2.83,   13.03,  0.33,  2.36, 0.516088, 2.369843),
    ("Baraff-Witkin", 20000, 0.04, 0.12, 13.77,  19.19,  1.79,  4.00, 0.057426, 0.170598),
    ("Stable Neo-Hookean", 79935, 0.19728, 0.59,    1.46,  216.93, 1.15, 10.70, 0.343491, 1.027440),
]

df = pd.DataFrame(
    data,
    columns=["Energy","N","Ours_G","Ours_H","Torch_G","Torch_H","JAX_G","JAX_H", "SymPy_G", "SymPy_H"]
)

# -----------------------------
# Convert to PER-INSTANCE time
# -----------------------------
for col in ["Ours_G","Ours_H","Torch_G","Torch_H","JAX_G","JAX_H", "SymPy_G", "SymPy_H"]:
    df[col] = df[col] / df["N"]

# -----------------------------
# Convert to slowdown (× Ours)
# -----------------------------
def make_relative_long(df, kind):  # kind in {"G","H"}
    rows = []
    for _, r in df.iterrows():
        ours = float(r[f"Ours_{kind}"])
        rows.append((r["Energy"], "Ours", 1.0))
        rows.append((r["Energy"], "PyTorch", float(r[f"Torch_{kind}"]) / ours))
        rows.append((r["Energy"], "JAX", float(r[f"JAX_{kind}"]) / ours))
        rows.append((r["Energy"], "SymPy", float(r[f"SymPy_{kind}"]) / ours))
    return pd.DataFrame(rows, columns=["Energy","Backend","Slowdown"])

df_long = pd.concat(
    [
        make_relative_long(df, "G").assign(Metric="Gradient"),
        make_relative_long(df, "H").assign(Metric="Hessian"),
    ],
    ignore_index=True
)

# -----------------------------
# Plot (horizontal bars, log-scale, centered legend)
# -----------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

g = sns.catplot(
    data=df_long,
    kind="bar",
    y="Energy",
    x="Slowdown",
    hue="Backend",
    col="Metric",
    orient="h",
    height=3.6,
    aspect=1.2,
    palette={
        "Ours": "#edae49",
        "PyTorch": "#d1495b",
        "JAX": "#00798c",
        "SymPy": "#6a4c93"
    },
    sharex=False
)

# Axis + titles
# g.set_axis_labels("Per-instance slowdown relative to Ours (×, log scale)", "")
g.set_titles("{col_name}")

# Log scale on x-axis
for ax in g.axes.flatten():
    ax.set_xscale("log")

# Center legend inside the figure
g._legend.set_title("")
g._legend.set_bbox_to_anchor((0.5, 0.5))
g._legend.set_loc("upper center")
g._legend.set_frame_on(False)

# Annotate bars
for ax in g.axes.flatten():
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1fx",
            label_type="edge",
            padding=3,
            fontsize=10
        )
    sns.despine(ax=ax, left=True, bottom=True)

for ax in g.axes.flatten():
    ax.set_xlabel("")
    ax.set_ylabel("")

for ax in g.axes.flatten():
    ax.grid(False)

# Add black outline to every bar
for ax in g.axes.flatten():
    for patch in ax.patches:
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

plt.tight_layout()
plt.show()
