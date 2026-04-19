import re
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({
  "font.family": "sans-serif",
  "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
})

def extract_time(file):
  global last_min_memory, last_v_count, last_max_memory
  total_hessian_time = 0.0
  total_cg_time = 0.0
  total_newton = 0
  total_cg_iterations = 0
  misc_time = 0.0
  cd_time = 0.0
  ccd_time = 0.0
  index_time = 0.0
  ccd_memory = 0.0
  max_memory = 0.0
  min_memory = 0.0
  average_memory = 0.0
  total_memory_lines = 0
  maximum_cd_pairs = 0
  num_v = 0
  num_f = 0
  num_e = 0

  time_re = re.compile(r"Time taken for (.*?): ([0-9.eE+-]+) seconds")
  total_re = re.compile(r"Total time:\s*([0-9.eE+-]+)")

  totals = defaultdict(float)
  counts = defaultdict(int)
  total_runtime = 0.0
  total_runtime_count = 0

  with open(file, "r") as f:
    for raw_line in f:
      line = raw_line.strip()

      if line.startswith("[hessianAndGradientKernel.compute] Finished in"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          total_hessian_time += float(m.group(1))

      elif line.startswith("Solver time: "):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          total_cg_time += float(m.group(1))

      elif line.startswith("Solver converged in"):
        m = re.search(r"(\d+)", line)
        if m:
          total_newton += 1
          total_cg_iterations += int(m.group(1))

      elif line.startswith("[globalKernel.compute]"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          misc_time += float(m.group(1))

      elif line.startswith("Continuous collision detection took"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          ccd_time += float(m.group(1))

      elif line.startswith("Collision detection took"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          cd_time += float(m.group(1))

      elif line.startswith("[gradientIndicesKernel.computeIndices]"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          index_time += float(m.group(1))

      elif line.startswith("Memory used by CCD"):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          ccd_memory += float(m.group(1))

      elif line.startswith("Memory used total: "):
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
        if m:
          max_memory = max(max_memory, float(m.group(1)))
          min_memory = min(min_memory, float(m.group(1))) if min_memory > 0 else float(m.group(1))
          average_memory += float(m.group(1))
          total_memory_lines += 1

      elif line.startswith("The separated counts for ccd are ["):
        nums_str = line.split("[", 1)[1].split("]", 1)[0]
        nums = [int(x.strip()) for x in nums_str.split(",") if x.strip()]
        maximum_cd_pairs = max(maximum_cd_pairs, sum(nums))

      elif line.startswith("Number of vertices "):
        m = re.search(r"(\d+)", line)
        if m:
          num_v = int(m.group(1))

      elif line.startswith("Number of triangles "):
        m = re.search(r"(\d+)", line)
        if m:
          num_f = int(m.group(1))

      elif line.startswith("Number of edges "):
        m = re.search(r"(\d+)", line)
        if m:
          num_e = int(m.group(1))

      m = time_re.match(line)
      if m:
        part = m.group(1)
        t = float(m.group(2))
        totals[part] += t
        counts[part] += 1
        continue

      m2 = total_re.match(line)
      if m2:
        t = float(m2.group(1))
        total_runtime += t
        total_runtime_count += 1


  result = {
    "file": file,
    "num_v": num_v,
    "num_f": num_f,
    "num_e": num_e,
    "maximum_cd_pairs": maximum_cd_pairs,
    "ccd_memory": ccd_memory,
    "yasps_memory_max": max_memory - ccd_memory,
    "yasps_memory_min": min_memory - ccd_memory,
    "yasps_memory_average": average_memory / total_memory_lines - ccd_memory if total_memory_lines > 0 else 0,
    "ccd_cd_total": (ccd_time + cd_time) / 1000.0,
    "diff_total": total_hessian_time / 1000.0,
    "cg_total": total_cg_time / 1000.0,
    "index_total": index_time / 1000.0,
    "misc_total": misc_time / 1000.0,
    "total_runtime": total_runtime,
    "total_newton": total_newton,
    "total_cg_iterations": total_cg_iterations,
    "part_totals": dict(totals),
    "part_counts": dict(counts),
    "total_runtime_count": total_runtime_count,
  }
  return result

counter = 6
def fmt_int(n):
  return f"{int(n):,}"


def fmt_float(x, digits=2):
  return f"{x:.{digits}f}"


def latex_row(stats):
  global counter
  return (
    f"{fmt_int(19193 * counter)} & "
    f"{fmt_int(counter)} & "
    f"{fmt_int(0)} & "
    f"False & "
    f"{fmt_float(stats['ccd_cd_total'], 2)} & "
    f"{fmt_float(stats['diff_total'], 2)} & "
    f"{fmt_float(stats['cg_total'], 2)} & "
    f"{fmt_float(stats['yasps_memory_max'], 2)} & "
    f"{fmt_float(10496 / 1204, 2)}\\\\ \\hline"
  )


def print_latex_table(stats_list):
  global counter
  # stats_list = [extract_time(file) for file in log_files]
  stats_list = sorted(stats_list, key=lambda s: s["num_v"])

  print(r"\begin{tabularx}{\textwidth}{|l|l|l|X|X|X|X|X|r|}")
  print(r"    \hline")
  print(r"    \thead{\# Vertices} & \thead{\# Soft\\Bunnies} & \thead{\# Affine\\Bunnies} &  \thead{Separate\\Jacobian}  & \thead{CCD \& CD\\Total(s)} & \thead{Diff\\Total(s)} & \thead{CG\\Total(s)} & \thead{Max YASPS\\Memory (MB)} & \thead{Thread Stack\\Limit(KB)} \\ \hline \hline")

  for stats in stats_list:
    print("    " + latex_row(stats))
    counter += 1

  print(r"\end{tabularx}")


def print_latex_rows_only(log_files):
  stats_list = [extract_time(file) for file in log_files]
  stats_list = sorted(stats_list, key=lambda s: s["num_v"])

  for stats in stats_list:
    print(latex_row(stats))


def plot_yasps_memory(stats_list):
  stats_list = sorted(stats_list, key=lambda s: s["num_v"])

  df = pd.DataFrame(stats_list[1:])

  plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
  })

  fig, ax = plt.subplots(
    1, 1,
    figsize=(3.75, 1.75),
    dpi=300,
    constrained_layout=True
  )

  ax.plot(
    df["num_v"],
    df["min_memory_per_10k"],
    linewidth=0.8,
    color="#E02B35",
    label = "Min YASPS Memory Needed per 10k Vertices"
  )

  ax.plot(
    df["num_v"],
    df["increased_memory_per_cp"],
    linewidth=0.8,
    color="#29A89C",
    label = "YASPS Memory Needed per 100k Collision Pairs"
  )

  ax.set_xlabel("Number of Vertices", labelpad=1, fontsize=6)
  ax.set_ylabel("Memory (MB)", labelpad=1, fontsize=6)

  ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
  ax.yaxis.get_offset_text().set_fontsize(6)
  ax.set_ylim(0, 60)

  ax.tick_params(axis="x", labelsize=6)
  ax.tick_params(axis="y", labelsize=6)

  ax.tick_params(axis="both", which="major", pad=1, length=2)
  ax.grid(False)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.legend(
    frameon=False,
    borderaxespad=0.1,
    handlelength=1.2,
    handletextpad=0.4,
    labelspacing=0.2,
    columnspacing=0.6,
    loc="lower right", fontsize=6
  )

  # plt.show()

  plt.savefig(
    "../plotting/mat_twist_memory.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.00
  )
  plt.close()


if __name__ == "__main__":
  log_files = [f"bunny_{i}.log" for i in range(6, 11)]

  stats = [extract_time(file) for file in log_files]

  print_latex_table(stats)
  # plot_yasps_memory(stats)
