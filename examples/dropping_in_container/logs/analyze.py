import re
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
COLORS = {
    "Differentiation+Assembly": "#081A34",              # blue
    "CG": "#E02B35",                   # orange
    "CCD+CD+LargestStep": "#29A89C",   # green
}


def extract_time(file, verbose=False):
  total_hessian_time = 0.0   # ms
  total_cg_time = 0.0        # ms
  total_newton = 0
  total_cg_iters = 0
  misc_time = 0.0            # ms
  cd_time = 0.0              # ms
  ccd_time = 0.0             # ms
  index_time = 0.0           # ms
  largest_step_time = 0.0    # ms
  separated_counts = np.array([0, 0, 0, 0])

  # -----------------------------
  # Pass 1: parse your log lines
  # -----------------------------
  line_number = 1
  with open(file, "r") as f:
    current_separate_counts = [0, 0, 0, 0]
    for line in f:
      if line.startswith("[energy.computeHessianAndGradient] Finished in"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        total_hessian_time += t
      elif line.startswith("Solver time: "):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        total_cg_time += t
      elif line.startswith("Solver converged in"):
        total_newton += 1
        cg_iterations = int(re.search(r"(\d+)", line).group(1))
        total_cg_iters += cg_iterations
      elif line.startswith("[globalKernel.compute]"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        misc_time += t
      elif line.startswith("Continuous collision detection took"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        ccd_time += t
      elif line.startswith("Collision detection took"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        cd_time += t
      elif line.startswith("Computing largest step size took"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        largest_step_time += t
      elif line.startswith("[gradientIndicesKernel.computeIndices]"):
        t = float(re.search(r"(\d+\.\d+)", line).group(1))
        index_time += t
      elif line.startswith("The separated counts are"):
        counts = list(map(int, re.findall(r"\d+", line)))
        if len(counts) == 4:
          current_separate_counts = counts
      elif line.startswith("=================================================================="):
        if len(current_separate_counts) == 4:
          separated_counts += np.array(current_separate_counts)
        current_separate_counts = [0, 0, 0, 0]
      line_number += 1



  # -----------------------------
  # Pass 2: parse "Time taken for X: Y seconds" + "Total time: Z"
  # -----------------------------
  time_re = re.compile(r"Time taken for (.*?): ([0-9.edge-edge+-]+) seconds")
  total_re = re.compile(r"Total time:\s*([0-9.edge-edge+-]+)")

  totals = defaultdict(float)
  counts = defaultdict(int)
  total_runtime = 0.0
  total_runtime_count = 0

  with open(file, "r") as f:
    for line in f:
      line = line.strip()

      m = time_re.match(line)
      if m:
        part = m.group(1)
        t = float(m.group(2))  # seconds
        totals[part] += t
        counts[part] += 1
        continue

      m2 = total_re.match(line)
      if m2:
        t = float(m2.group(1))
        total_runtime += t
        total_runtime_count += 1

  # Your main buckets (seconds)
  totals["Differentiation+Assembly"] = total_hessian_time / 1000.0
  totals["CG"] = total_cg_time / 1000.0
  totals["CCD+CD+LargestStep"] = (ccd_time + cd_time + largest_step_time) / 1000.0
  totals["point-point"] = separated_counts[0]
  totals["point-edge"] = separated_counts[1]
  totals["point-triangle"] = separated_counts[2]
  totals["edge-edge"] = separated_counts[3]

  # Keep if you still want it
  totals["total"] = total_runtime

  if verbose:
    print("Total Hessian time (s):", totals["Differentiation+Assembly"])
    print("Total CG time (s):", totals["CG"])
    print("Total CCD+CD+LargestStep (s):", totals["CCD+CD+LargestStep"])
    if total_runtime_count > 0:
      print("Total time sum (s):", total_runtime)

  return totals


if __name__ == "__main__":
  rows = []
  extract_time(f"bunny_5.log", verbose=True)
  extract_time(f"bunny_5_new.log", verbose=True)
  # for i in range(25):
  #   bunny_id = i + 1
  #   t = extract_time(f"bunny_{bunny_id}.log", verbose=False)
  #   rows.append({
  #     "bunnies": bunny_id,
  #     "Differentiation+Assembly": t["Differentiation+Assembly"],
  #     "CG": t["CG"],
  #     "CCD+CD+LargestStep": t["CCD+CD+LargestStep"],

  #     "point-point": t["point-point"],
  #     "point-edge": t["point-edge"],
  #     "point-triangle": t["point-triangle"],
  #     "edge-edge": t["edge-edge"],
  #   })

  # df = pd.DataFrame(rows).sort_values("bunnies")

  # # =============================
  # # Two plots: top (runtime), bottom (collision counts)
  # # =============================
  # fig, (ax1, ax2) = plt.subplots(
  #     2, 1,
  #     figsize=(3.75, 3.0),   # double height
  #     dpi=300,
  #     # sharex=True,
  #     gridspec_kw={"hspace": 0.1},
  #     constrained_layout=True
  # )

  # plt.rcParams.update({
  #     "font.size": 6,
  #     "axes.labelsize": 6,
  #     "axes.titlesize": 6,
  #     "xtick.labelsize": 6,
  #     "ytick.labelsize": 6,
  #     "legend.fontsize": 6,
  # })

  # # -----------------------------
  # # Top plot: runtime
  # # -----------------------------
  # ax1.plot(df["bunnies"], df["Differentiation+Assembly"],
  #          label="Differentiation+Assembly",
  #          color=COLORS["Differentiation+Assembly"], linewidth=1)
  # ax1.plot(df["bunnies"], df["CG"],
  #          label="CG", color=COLORS["CG"], linewidth=1)
  # ax1.plot(df["bunnies"], df["CCD+CD+LargestStep"],
  #          label="CCD+CD+LargestStep",
  #          color=COLORS["CCD+CD+LargestStep"], linewidth=1)

  # ax1.set_ylabel("Time (s)", labelpad=1, fontsize=6)
  # ax1.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
  # ax1.set_xticks([5, 10, 15, 20, 25])
  # ax1.tick_params(axis="x", labelsize=6)
  # ax1.tick_params(axis="y", labelsize=6)
  # ax1.legend(
  #     frameon=False,
  #     borderaxespad=0.1,
  #     handlelength=1.2,
  #     handletextpad=0.4,
  #     labelspacing=0.2,
  #     columnspacing=0.6,
  #     loc="upper left", fontsize=6
  # )

  # # -----------------------------
  # # Bottom plot: collision counts
  # # -----------------------------
  # COUNT_COLORS = {
  #     "point-point": "#26547c",
  #     "point-edge": "#ef476f",
  #     "point-triangle": "#ffd166",
  #     "edge-edge": "#06d6a0",
  # }

  # ax2.plot(df["bunnies"], df["point-point"],
  #          label="point-point", color=COUNT_COLORS["point-point"], linewidth=1)
  # ax2.plot(df["bunnies"], df["point-edge"],
  #          label="point-edge", color=COUNT_COLORS["point-edge"], linewidth=1)
  # ax2.plot(df["bunnies"], df["point-triangle"],
  #          label="point-triangle", color=COUNT_COLORS["point-triangle"], linewidth=1)
  # ax2.plot(df["bunnies"], df["edge-edge"],
  #          label="edge-edge", color=COUNT_COLORS["edge-edge"], linewidth=1)

  # ax2.set_xlabel("Number of bunnies", labelpad=1, fontsize=6)
  # ax2.set_ylabel("Count", labelpad=1, fontsize=6)
  # ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
  # ax2.yaxis.get_offset_text().set_fontsize(6)


  # ax2.set_xticks([5, 10, 15, 20, 25])
  # ax2.tick_params(axis="x", labelsize=6)
  # ax2.tick_params(axis="y", labelsize=6)

  # ax2.legend(
  #     frameon=False,
  #     borderaxespad=0.1,
  #     handlelength=1.2,
  #     handletextpad=0.4,
  #     labelspacing=0.2,
  #     columnspacing=0.6,
  #     loc="upper left", fontsize=6
  # )

  # # -----------------------------
  # # Shared formatting
  # # -----------------------------
  # for ax in (ax1, ax2):
  #     ax.set_xlim(1, 25)
  #     ax.set_xticks([5, 10, 15, 20, 25])
  #     ax.tick_params(axis="both", which="major", pad=1, length=2)
  #     ax.grid(False)
  #     ax.spines["top"].set_visible(False)
  #     ax.spines["right"].set_visible(False)

  # # plt.show()
  # plt.savefig(
  #   "../plotting/scalability.pdf",
  #   format="pdf",
  #   bbox_inches="tight",
  #   pad_inches=0.00
  # )
  # plt.close()
