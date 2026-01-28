import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------
# Global style (publication grade)
# --------------------------------
sns.set_theme(
    context="paper",
    style="whitegrid",
    font_scale=1.4,
)

# --------------------------------
# Known sizes + ours
# --------------------------------
BASE_BLOCKS_PER_N = 1
BS = 3
ns = list(range(1, 21))

matrix_size = np.array(ns)

time_ours = np.array([
  0.017756,
  0.029551,
  0.038435,
  0.048620,
  0.071788,
  0.080159,
  0.105964,
  0.107802,
  0.115241,
  0.127311,
  0.139428,
  0.151101,
  0.163643,
  0.176225,
  0.187843,
  0.200041,
  0.211981,
  0.223724,
  0.236257,
  0.248276,
])

# --------------------------------
# Parse benchmark output
# --------------------------------
with open("bench_output.txt", "r") as f:
    text = f.read()

start_re = re.compile(r"hessian_blocks_(\d+)_(full|upper)_(BCOO|BCSR|COO|CSR)\.bin")
time_re  = re.compile(r"Full-SpMV time:\s*([0-9.]+)\s*ms")

records = []
current = None

for line in text.splitlines():
    m = start_re.search(line)
    if m:
        current = (int(m.group(1)), m.group(2), m.group(3))
        continue

    m = time_re.search(line)
    if m and current:
        n, variant, fmt = current
        records.append({
            "n": n,
            "matrix_size": matrix_size[n - 1],
            "time_ms": float(m.group(1)),
            "Format": fmt,
            "Storage": variant,
            "method": f"{variant}_{fmt}"
        })
        current = None

df = pd.DataFrame(records)

# Add "ours"
df_ours = pd.DataFrame({
    "n": ns,
    "matrix_size": matrix_size,
    "time_ms": time_ours,
    "Format": ["Ours"] * len(ns),
    "Storage": ["upper"] * len(ns),
    "method": ["Ours"] * len(ns),
})

df = pd.concat([df, df_ours], ignore_index=True)

# --------------------------------
# Color & style mapping (carefully chosen)
# --------------------------------
palette = {
    "BCOO": "#1f77b4",   # blue
    "BCSR": "#ff7f0e",   # orange
    "COO":  "#2ca02c",   # green
    "CSR":  "#d62728",   # red
    "Ours": "#000000",   # black
}

style_order = ["full", "upper"]

# --------------------------------
# Plot (SINGLE FIGURE)
# --------------------------------
plt.figure(figsize=(11, 6.5), dpi=220)

ax = sns.lineplot(
    data=df,
    x="n",
    y="time_ms",
    hue="Format",
    style="Storage",
    style_order=style_order,
    palette=palette,
    linewidth=2.2,
    markers=True,
    dashes={"full": "", "upper": (4, 2)},
)

ax.set_title("SpMV Performance Comparison (All Formats)")
ax.set_xlabel("Matrix size M (scalar dimension)")
ax.set_ylabel("Average SpMV time (ms)")

# Clean x-axis labels
ax.ticklabel_format(style="plain", axis="x")
ax.grid(False)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks(range(2, 21, 2))
ax.set_xlim(0, 21)
# Legend formatting
ax.legend(
    loc="upper left",
    frameon=False,
)
sns.set_theme(
    context="paper",
    style="white",   # ← instead of "whitegrid"
    font_scale=1.4,
)
ax.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
# plt.savefig("spmv_all_methods_seaborn.png", bbox_inches="tight")
# plt.show()
plt.savefig(
  "../../plotting/spmv_all_methods_seaborn..pdf",
  format="pdf",
  bbox_inches="tight",
  pad_inches=0.00
)
plt.close()
