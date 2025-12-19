import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

parts = [
    "Setup",
    "YASPS Integration",
    "Energy Definition",
    "Collision Detection Initialization",
    "Visualization",
    "Newton Iteration"
]

simulation_names = [
    "Soft + Cloth",
    "Soft + Cloth + ABD",
    "Soft + Cloth + ABD + Cage"
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
df_long = df.reset_index().melt(
    id_vars="index",
    var_name="Stage",
    value_name="Lines of Code"
)
df_long.rename(columns={"index": "Simulation"}, inplace=True)

# palette = sns.color_palette("muted", n_colors=6)
custom_colors = {
    "Setup": "#ef476f",                     # muted blue
    "YASPS Integration": "#f78c6b",          # muted green
    "Energy Definition": "#ffd166",          # muted red
    "Collision Detection Initialization": "#06d6a0",  # purple
    "Visualization": "#118ab2",              # warm yellow
    "Newton Iteration": "#073b4c"             # cyan
}
sns.set_theme(style="whitegrid", font_scale=1.15)

fig, ax = plt.subplots(figsize=(12, 6))
ax.grid(False)
sns.barplot(
    data=df_long,
    x="Simulation",
    y="Lines of Code",
    hue="Stage",
    palette=custom_colors,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

# ax.set_title("Lines of Code by Simulation Stage")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticks([])

ax.legend(
  loc="upper left",      # inside the axes
  ncol=2,                 # two columns
  frameon=False,           # or False if you want no box
  fontsize=11,
  title_fontsize=11
)
# Add value labels on top of each bar
for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%d',
        padding=1,
        fontsize=10,
        fontweight='semibold'
    )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.show()
