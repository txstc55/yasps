import re
from collections import defaultdict
# f = open("one_bunny_2_cloth_soft.log")


def extract_time(file):
  f = open(file)
  total_hessian_time = 0.0
  total_cg_time = 0.0
  total_newton = 0
  total_cg = 0
  misc_time = 0
  cd_time = 0.0
  ccd_time = 0.0
  index_time = 0.0

  for line in f:
    if line.startswith("[energy.computeHessianAndGradient] Finished in"):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      total_hessian_time += time
    elif line.startswith("Solver time: "):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      total_cg_time += time
      total_cg += 1
    elif line.startswith("Solver converged in"):
      total_newton += 1
      cg_iterations = int(re.search(r"(\d+)", line).group(1))
      total_cg += cg_iterations
    elif line.startswith("[globalKernel.compute]"):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      misc_time += time
    elif line.startswith("Continuous collision detection took"):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      ccd_time += time
    elif line.startswith("Collision detection took"):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      cd_time += time
    elif line.startswith("[gradientIndicesKernel.computeIndices]"):
      time = float(re.search(r"(\d+\.\d+)", line).group(1))
      index_time += time
  f.close()
  print("Total Hessian time:", total_hessian_time / 1000.0)
  print("Total CG time:", total_cg_time / 1000.0)
  print("Total Newton iterations:", total_newton)
  print("Total CG iterations:", total_cg)
  print("Total compute time:", misc_time / 1000.0)
  print("Total CD time:", cd_time / 1000.0)
  print("Total CCD time:", ccd_time / 1000.0)
  print("Total Index time:", index_time / 1000.0)



  # Regex for the "Time taken for X: Y seconds" lines
  time_re = re.compile(r"Time taken for (.*?): ([0-9.eE+-]+) seconds")

  # Regex for the overall "Total time: Z" (if you care about it)
  total_re = re.compile(r"Total time:\s*([0-9.eE+-]+)")

  # Accumulators
  totals = defaultdict(float)
  counts = defaultdict(int)
  total_runtime = 0.0
  total_runtime_count = 0

  with open(file, "r") as f:
    for line in f:
      line = line.strip()

      # Match per-part timings
      m = time_re.match(line)
      if m:
        part = m.group(1)              # e.g. "solver", "data transfer"
        t = float(m.group(2))          # seconds
        totals[part] += t
        counts[part] += 1
        continue

      # Match overall total time
      m2 = total_re.match(line)
      if m2:
        t = float(m2.group(1))
        total_runtime += t
        total_runtime_count += 1

  print("=== Per-part timing totals ===")
  for part in sorted(totals.keys()):
    total_t = totals[part]
    n = counts[part]
    avg = total_t / n if n > 0 else 0.0
    print(f"{part:25s} total = {total_t:10.6f} s  (calls: {n:4d}, avg: {avg:.6f} s)")

  if total_runtime_count > 0:
    print("\n=== Overall run time lines ===")
    print(f"Total time sum   = {total_runtime:.6f} s")
    print(f"Total time count = {total_runtime_count}")
    print(f"Average total    = {total_runtime / total_runtime_count:.6f} s")

  totals["Hessian"] = total_hessian_time / 1000.0
  totals["CG"] = total_cg_time / 1000.0
  totals["total"] = total_runtime
  return totals


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})

def plot_time_distributions(**named_dicts):
    """
    Stacked horizontal bars in absolute time, with percentage labels
    on Differentiation+Assembly, CG, and CCD+CD+LargestStep.
    """

    sns.set_theme(style="white")

    renamed_hessian = "Differentiation+Assembly"

    categories = [
        renamed_hessian,
        "CG",
        "Computation",
        "CCD+CD+LargestStep",
        "Data Transfer",
        "Misc"
    ]

    # Custom colors
    colors = {
        renamed_hessian: "#081A34",
        "CG": "#E02B35",
        "Computation": "#F0C571",
        "CCD+CD+LargestStep": "#29A89C",
        "Data Transfer": "#A559AA",
        "Misc": "#CECECE",
    }

    label_categories = {
        renamed_hessian,
        "CG",
        "CCD+CD+LargestStep",
    }

    def collapse(d):
        total = d["total"]

        hessian = d.get("Hessian", 0)
        cg = d.get("CG", 0)
        computation = d.get("computation", 0)
        communication = (
            d.get("CCD", 0)
            + d.get("collision detection", 0)
            + d.get("largest step", 0)
        )
        data_transfer = d.get("data transfer", 0)

        misc = total - (hessian + cg + communication + data_transfer + computation)

        collapsed = {
            renamed_hessian: hessian,
            "CG": cg,
            "Computation": computation,
            "CCD+CD+LargestStep": communication,
            "Data Transfer": data_transfer,
            "Misc": misc,
            "total": total,
        }

        return collapsed

    # Build DataFrame
    df = pd.DataFrame(
        {run_name: collapse(d) for run_name, d in named_dicts.items()}
    ).T

    plt.figure(figsize=(3.75, 1.5), dpi=300)
    ax = plt.gca()

    left = [0.0] * len(df)

    # Draw stacked bars
    for cat in categories:
        bar_container = ax.barh(
            df.index,
            df[cat],
            left=left,
            label=cat,
            color=colors[cat],
            edgecolor="black",
            linewidth=0.4
        )

        # Add percentage label INSIDE the bar for selected categories
        if cat in label_categories:
            for i, rect in enumerate(bar_container.patches):
                width = rect.get_width()
                total = df.iloc[i]["total"]
                pct = width / total * 100 if total > 0 else 0

                if width > 0:  # Only label if segment is non-zero
                    # Do not label tiny bars (avoid clutter)
                    if pct >= 3:
                        x_center = rect.get_x() + width / 2
                        y_center = rect.get_y() + rect.get_height() / 2
                        ax.text(
                            x_center,
                            y_center,
                            f"{pct:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="white",
                            fontweight="bold"
                        )

        # Update stacking offset
        left = [l + v for l, v in zip(left, df[cat])]

    # Remove grid and spines
    ax.grid(False)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    # No numeric ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend inside
    ax.legend(
        title="",
        loc="lower right",
        frameon=False,
        fontsize=5,
        bbox_to_anchor=(1.02, 0.0),
    )

    plt.tight_layout()
    plt.show()









if __name__ == "__main__":
  times2 = extract_time("one_bunny_2_cloth_soft.log")
  times3 = extract_time("one_bunny_3_cloth_soft.log")
  times4 = extract_time("one_bunny_4_cloth_soft.log")

  # plot_time_distributions(time_2=times2, time_3=times3, time_4=times4)
  # print(times)
