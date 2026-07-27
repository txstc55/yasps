#!/usr/bin/env python3

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FRAMES = 24

VARIANTS = {
  "one_bunny_partial_abd": {
    "description": "Partial ABD with combined Jacobian/Hessian",
    "configuration": "1 bunny",
    "video": "evaluation/one_bunny_partial_abd.mp4",
  },
  "one_bunny_partial_abd_separate_jacobian": {
    "description": "Partial ABD with separate Jacobian",
    "configuration": "1 bunny",
    "video": (
      "evaluation/"
      "one_bunny_partial_abd_separate_jacobian.mp4"
    ),
  },
  "dropping_in_container": {
    "description": "Soft bunny dropped in a container",
    "configuration": "1 soft bunny",
    "video": "evaluation/dropping_in_container.mp4",
  },
  "dropping_in_container_mixed": {
    "description": "Mixed soft/affine container drop",
    "configuration": "1 soft + 1 affine bunny",
    "video": "evaluation/dropping_in_container_mixed.mp4",
  },
  "dropping_in_container_mixed_separation": {
    "description": "Mixed container drop with separation",
    "configuration": "1 soft + 1 affine bunny",
    "video": (
      "evaluation/"
      "dropping_in_container_mixed_separation.mp4"
    ),
  },
  "dropping_in_container_no_save": {
    "description": "Soft container drop without rendering",
    "configuration": "1 soft bunny",
    "video": None,
  },
}

INDEX_KERNELS = {
  "bitonic_coord_dim_step_metal",
  "bitonic_gradient_size_step_metal",
  "build_coordinate_metadata_metal",
  "compute_coordinates_metal",
  "compute_permutation_metal",
  "copy_uint_metal",
  "extract_unique_coordinates_metal",
  "group_gradient_sizes_metal",
  "inclusive_scan_uint_step_metal",
  "lookup_coordinate_offsets_metal",
  "pack_coord_dims_metal",
  "reorderPlacementIndicesGlobal",
  "unique_coord_dims_metal",
}

CCD_KERNELS = {
  "bitonic_morton_step",
  "calculate_edge_leaf_boxes",
  "calculate_face_leaf_boxes",
  "calculate_internal_boxes_independent",
  "calculate_internal_nodes",
  "calculate_leaf_nodes",
  "calculate_morton_hashes",
  "calculate_step_reciprocals",
  "copy_aabb",
  "fill_uint",
  "query_edges_ccd",
  "query_edges_cd",
  "query_faces_ccd",
  "query_faces_cd",
  "reduce_aabbs",
  "reduce_max_float",
  "separate_edge_pairs",
  "separate_face_pairs",
  "sort_leaf_boxes",
}

STAGES = (
  "fused_compute",
  "sparse_indices",
  "hessian_assembly",
  "linear_solver",
  "ccd",
  "array_runtime",
)


def classify_kernel(name):
  if name.startswith("yasps_"):
    return "array_runtime"
  if name.startswith("compute_hessian_and_gradient"):
    return "hessian_assembly"
  if (
    name.startswith((
      "spmv_blocks_",
      "block_jacobi_",
      "dot_product_",
      "sum_partial_",
      "vec_add_with_scalar_",
      "invert_diagonal_blocks_",
    ))
    or name == "fill_float_metal"
  ):
    return "linear_solver"
  if name.endswith("_get_indices_metal") or name in INDEX_KERNELS:
    return "sparse_indices"
  if name.endswith("_global_function"):
    return "fused_compute"
  if name in CCD_KERNELS:
    return "ccd"
  raise ValueError(f"Unclassified Metal kernel: {name}")


def extract_numbers(pattern, text, cast=float):
  return [
    cast(value)
    for value in re.findall(pattern, text, re.MULTILINE)
  ]


def summarize_variant(name, metadata):
  directory = ROOT / name / "evaluation"
  timing = json.loads(
    (directory / "kernel_timings.json").read_text()
  )
  log = (directory / "run.log").read_text(errors="replace")

  stages = {stage: 0.0 for stage in STAGES}
  kernel_calls = 0
  dispatch_wall_ms = 0.0
  for kernel_name, values in timing["kernels"].items():
    stages[classify_kernel(kernel_name)] += values["gpu_ms"]
    kernel_calls += values["calls"]
    dispatch_wall_ms += values["wall_ms"]

  wall_seconds = extract_numbers(
    r"^real ([0-9.]+)$",
    log,
  )[-1]
  cg_iterations = extract_numbers(
    r"Solver converged in (\d+) iterations",
    log,
    int,
  )
  line_search_substeps = extract_numbers(
    r"^substep is (\d+)$",
    log,
    int,
  )
  collision_candidates = extract_numbers(
    r"number of collision pairs: (\d+)",
    log,
    int,
  )
  frame_directory = directory / "frames"
  frame_count = (
    len(list(frame_directory.glob("frame_*.png")))
    if frame_directory.exists()
    else FRAMES
  )
  total_gpu_ms = sum(stages.values())

  result = {
    **metadata,
    "device": timing["device"],
    "frames": frame_count,
    "wall_seconds": wall_seconds,
    "kernel_gpu_ms": total_gpu_ms,
    "kernel_gpu_ms_per_frame": total_gpu_ms / frame_count,
    "dispatch_wall_ms": dispatch_wall_ms,
    "kernel_calls": kernel_calls,
    "nonlinear_solves": len(cg_iterations),
    "cg_iterations": {
      "total": sum(cg_iterations),
      "mean": (
        sum(cg_iterations) / len(cg_iterations)
        if cg_iterations
        else 0.0
      ),
      "minimum": min(cg_iterations, default=0),
      "maximum": max(cg_iterations, default=0),
    },
    "line_search_substeps": {
      "mean": (
        sum(line_search_substeps) / len(line_search_substeps)
        if line_search_substeps
        else 0.0
      ),
      "maximum": max(line_search_substeps, default=0),
    },
    "collision_candidates": {
      "mean": (
        sum(collision_candidates) / len(collision_candidates)
        if collision_candidates
        else 0.0
      ),
      "maximum": max(collision_candidates, default=0),
    },
    "stage_gpu_ms": stages,
    "top_kernels": [
      {
        "name": kernel_name,
        "calls": values["calls"],
        "gpu_ms": values["gpu_ms"],
      }
      for kernel_name, values in sorted(
        timing["kernels"].items(),
        key=lambda item: item[1]["gpu_ms"],
        reverse=True,
      )[:10]
    ],
  }
  return result


def write_csv(summary):
  fields = [
    "variant",
    "configuration",
    "frames",
    "wall_seconds",
    "kernel_gpu_ms",
    "kernel_gpu_ms_per_frame",
    "kernel_calls",
    "nonlinear_solves",
    "cg_iterations_total",
    "cg_iterations_mean",
    *[f"{stage}_gpu_ms" for stage in STAGES],
  ]
  with (ROOT / "summary.csv").open("w", newline="") as output:
    writer = csv.DictWriter(
      output,
      fieldnames=fields,
      lineterminator="\n",
    )
    writer.writeheader()
    for name, result in summary["variants"].items():
      row = {
        "variant": name,
        "configuration": result["configuration"],
        "frames": result["frames"],
        "wall_seconds": result["wall_seconds"],
        "kernel_gpu_ms": result["kernel_gpu_ms"],
        "kernel_gpu_ms_per_frame": (
          result["kernel_gpu_ms_per_frame"]
        ),
        "kernel_calls": result["kernel_calls"],
        "nonlinear_solves": result["nonlinear_solves"],
        "cg_iterations_total": result["cg_iterations"]["total"],
        "cg_iterations_mean": result["cg_iterations"]["mean"],
      }
      row.update({
        f"{stage}_gpu_ms": result["stage_gpu_ms"][stage]
        for stage in STAGES
      })
      writer.writerow(row)


def main():
  variants = {
    name: summarize_variant(name, metadata)
    for name, metadata in VARIANTS.items()
  }
  devices = {result["device"] for result in variants.values()}
  if len(devices) != 1:
    raise ValueError(f"Expected one Metal device, got {devices}")
  summary = {
    "evaluated_on": "2026-07-26",
    "backend": "metal",
    "device": devices.pop(),
    "dtype": "float32",
    "policy": {
      "frames": FRAMES,
      "render_resolution": [960, 540],
      "video_fps": 12,
      "shader_cache": "warm",
      "timing_scope": (
        "MTLCommandBuffer GPU time; compilation and rendering excluded"
      ),
      "nonlinear_stopping": "example defaults; no iteration cap",
    },
    "variants": variants,
  }
  (ROOT / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
  )
  write_csv(summary)


if __name__ == "__main__":
  main()
