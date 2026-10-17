"""Package-level regression for direct-lower-input EVD with PSD checks bypassed.

CPU source/cache checks: python yasps/tests/test_small_evd_integration.py --cpu-only
GPU integration: PYTHONPATH=yasps /usr/bin/python3 yasps/tests/test_small_evd_integration.py --output /tmp/evd-integration
All generated CUDA/cache files belong to the supplied fresh output directory.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "yasps/yasps"
PROJECTION = PACKAGE / "projection"
HEADER_HASH = "3331ce78cb573eee3ed4d6da4b366ee4ce078f5af7d2dfa5950670da0e8fbbd2"
HELPERS = {"compute": PACKAGE / "kernel/Compute/globalKernel.pyx",
           "hessian": PACKAGE / "kernel/Hessian/hessianKernelHeader.pyx"}
SIZES = (1, 2, 3, 4, 6, 9, 12, 16)


def literal_header(source):
  return next(node.value for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and "void spd_projection_inplace" in node.value)


def source_checks():
  # Load the pure-Python packaging helper without importing CUDA-backed YASPS.
  spec = importlib.util.spec_from_file_location("projection_under_test", PROJECTION / "__init__.py")
  projection = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(projection)
  header = (PROJECTION / "small_evd.cuh").read_bytes()
  assert hashlib.sha256(header).hexdigest() == HEADER_HASH
  assert projection.SMALL_EVD_HEADER == header.decode()
  tag = projection.PROJECTION_CACHE_TAG
  assert tag == "evd_reuse_direct_lower_v2_" + HEADER_HASH[:12]
  assert "Mozilla Public" in projection.SMALL_EVD_HEADER
  assert (PROJECTION / "LICENSE.txt").read_text().startswith("Mozilla Public License Version 2.0")
  helpers = {}
  for kind, path in HELPERS.items():
    source = path.read_text()
    helper = literal_header(source)
    for name in ("is_positive_semidefinite", "is_positive_semidefinite_reuse_upper"):
      assert re.search(r"bool\s+" + name + r"\s*\(", helper), name
    assert not re.search(r"\bis_positive_semidefinite(?:_reuse_upper)?\s*<", helper)
    assert helper.count("small_evd::project_reuse_output<N, true>") == 2
    direct = helper.split("void spd_projection(", 1)[1]
    assert "project_reuse_output<N, true>(A, output, choice)" in direct
    # Only method 0 may copy. Nonzero projection must read the input directly.
    assert "for (" not in direct.split("return;", 1)[1]
    assert helper.count('asm volatile("trap;")') == 2
    small = helper.split("void spd_projection_small(", 1)[1].split("\ntemplate", 1)[0]
    assert "Eigen::SelfAdjointEigenSolver" in small  # Legacy small path is intentional.
    assert "A[0] < 1e-6 ? 1e-6" in small
    assert (": 1e-6;" if kind == "compute" else ": 0.0;") in small
    bypass = small.split("if (choice == 0)", 1)[1].split("if (N == 1)", 1)[0]
    assert ("output[i] = A[i]" in bypass) == (kind == "compute")
    assert "SMALL_EVD_HEADER" in source
    helpers[kind] = projection.SMALL_EVD_HEADER + helper
  # All native objects that may contain inlined helpers must change cache key.
  checks = ((HELPERS["compute"], ("file_name", "obj_file")),
            (PACKAGE / "kernel/Hessian/hessianAndGradientKernel.pyx", ("full_file_name", "obj_file")),
            (PACKAGE / "kernel/Jacobian/secondOrderJacobianKernel.pyx", ("signature", "object_file")))
  for path, names in checks:
    source = path.read_text()
    assignments = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Assign, ast.AugAssign))]
    for name in names:
      values = [ast.get_source_segment(source, node.value) for node in assignments
                if any(isinstance(target, ast.Name) and target.id == name
                       for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
      assert any("PROJECTION_CACHE_TAG" in value for value in values), (path, name)
    for signature, hashed in (("full_file_name", "full_file_name_hashed"), ("signature", "signature_hash")):
      updates = [node.lineno for node in assignments
                 if any(isinstance(target, ast.Name) and target.id == signature
                        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
                 and "PROJECTION_CACHE_TAG" in ast.get_source_segment(source, node.value)]
      hashes = [node.lineno for node in assignments
                if any(isinstance(target, ast.Name) and target.id == hashed
                       for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
      if hashes:
        assert updates and max(updates) < min(hashes), (path, "tag must precede hash")
  setup_tree = ast.parse((ROOT / "yasps/setup.py").read_text())
  setup_call = next(node for node in ast.walk(setup_tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "setup")
  package_data = ast.literal_eval(next(kw.value for kw in setup_call.keywords if kw.arg == "package_data"))
  assert {"*.cuh", "LICENSE.txt"} <= set(package_data["yasps.projection"])
  return projection, helpers


def matrices(n):
  rng = np.random.default_rng(953 + n)
  raw = rng.normal(size=(12, n, n))
  a = (raw + raw.transpose(0, 2, 1)) * .5
  return np.ascontiguousarray(np.concatenate([a, a * 1e-100, a * 1e100,
    [np.zeros((n, n)), np.eye(n), -np.eye(n), raw[0] @ raw[0].T,
     np.diag(np.r_[-1e-14, np.ones(n-1)])]]))


def expected_projection(a, choice, negative_floor=0.):
  if choice == 0:
    return a.copy()
  scale = np.max(np.abs(a), axis=(-2, -1))
  scale = np.where(scale == 0., 1., scale)
  values, vectors = np.linalg.eigh(a / scale[..., None, None])
  values = values * scale[..., None]
  values = np.abs(values) if choice == 1 else np.where(values < 0., negative_floor, values)
  return (vectors * values[..., None, :]) @ np.swapaxes(vectors, -2, -1)


def helper_checks(helpers, output):
  import pycuda.autoinit
  import pycuda.gpuarray as gpuarray
  from pycuda.compiler import SourceModule
  rows = []
  for kind, helper in helpers.items():
    wrappers = []
    for n in SIZES:
      modes = ("copy", "alias", "inplace", "legacy") + (("small",) if n <= 3 else ())
      for mode in modes:
        call = (f"spd_projection_small<{n}>(input+tid*{n*n}, a, choice);" if mode == "small" else
                f"spd_projection<{n}>(input+tid*{n*n}, a, choice);" if mode == "copy" else
                f"spd_projection<{n}>(a, a, choice);" if mode == "alias" else
                f"spd_projection_inplace<{n}>(a, choice);")
        if mode == "legacy":
          # Previous wrapper: separate mirror, then column-major decomposition.
          call = f'''if (choice != 0) {{
            for (int row=1; row<{n}; ++row)
              for (int col=0; col<row; ++col) a[col*{n}+row]=a[row*{n}+col];
            if (!small_evd::project_reuse_output<{n}>(a, a, choice)) asm volatile("trap;");
          }}'''
        initialize = "input[tid*%d+i]" % (n*n) if mode in ("alias", "inplace", "legacy") else "12345."
        wrappers.append(f'''extern "C" __global__ void project_{n}_{mode}(const double *input, double *output, int choice, int count) {{
          int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=count) return;
          double a[{n*n}]; for(int i=0;i<{n*n};++i) a[i]={initialize};
          {call}
          for(int i=0;i<{n*n};++i) output[tid*{n*n}+i]=a[i];
        }}''')
    directory = output / "helpers" / kind
    directory.mkdir(parents=True)
    code = helper + "\n" + "\n".join(wrappers)
    (directory / "helpers.cu").write_text(code)
    module = SourceModule(code, no_extern_c=True, include_dirs=["/usr/include/eigen3"],
      options=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--disable-warnings"], cache_dir=str(directory))
    maximum = 0.
    for n in SIZES:
      modes = ("copy", "alias", "inplace") + (("small",) if n <= 3 else ())
      for mode in modes:
        inputs = (np.asarray([np.eye(n), -np.eye(n), np.eye(n)*1e-10])
                  if mode == "small" else matrices(n))
        original = gpuarray.to_gpu(inputs)
        result = gpuarray.empty_like(original)
        legacy = gpuarray.empty_like(original)
        for choice in (0, 1, 2):
          module.get_function(f"project_{n}_{mode}")(original, result, np.int32(choice), np.int32(len(inputs)),
            block=(32, 1, 1), grid=((len(inputs)+31)//32, 1, 1))
          actual = result.get()
          if mode != "small":
            module.get_function(f"project_{n}_legacy")(original, legacy, np.int32(choice), np.int32(len(inputs)),
              block=(32, 1, 1), grid=((len(inputs)+31)//32, 1, 1))
            np.testing.assert_array_equal(actual.view(np.uint64), legacy.get().view(np.uint64))
          expected = expected_projection(inputs, choice, 1e-6 if mode == "small" and kind == "compute" else 0.)
          if mode == "small" and n == 1 and choice == 2:
            expected = np.maximum(inputs, 1e-6)
          if mode == "small" and kind == "hessian" and choice == 0:
            expected = np.full_like(inputs, 12345.)
          scale = np.maximum(np.max(np.abs(inputs), axis=(1, 2)), 1e-100)
          error = np.max(np.abs(actual-expected), axis=(1, 2)) / scale
          maximum = max(maximum, float(error.max()))
          assert np.isfinite(actual).all() and error.max() < 2e-11, (kind, n, mode, choice, error.max())
          if choice:
            np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
            assert np.linalg.eigvalsh(actual / scale[:, None, None]).min() > -2e-11
          np.testing.assert_array_equal(original.get().view(np.uint64), inputs.view(np.uint64))
      if n > 1:
        # General helpers consume row-major lower storage, even when upper is junk.
        inputs = matrices(n)[:4].copy()
        upper = np.triu_indices(n, 1)
        inputs[:2, upper[0], upper[1]] = 12345.
        inputs[2:, upper[0], upper[1]] = np.nan
        lower = np.tril(inputs)
        symmetric = lower + np.swapaxes(np.tril(inputs, -1), -2, -1)
        original = gpuarray.to_gpu(inputs)
        result = gpuarray.empty_like(original)
        legacy = gpuarray.empty_like(original)
        scale = np.max(np.abs(symmetric), axis=(1, 2))
        for mode in ("copy", "alias", "inplace"):
          for choice in (1, 2):
            module.get_function(f"project_{n}_{mode}")(original, result, np.int32(choice), np.int32(len(inputs)),
              block=(32, 1, 1), grid=(1, 1, 1))
            actual = result.get()
            module.get_function(f"project_{n}_legacy")(original, legacy, np.int32(choice), np.int32(len(inputs)),
              block=(32, 1, 1), grid=(1, 1, 1))
            np.testing.assert_array_equal(actual.view(np.uint64), legacy.get().view(np.uint64))
            error = np.max(np.abs(actual-expected_projection(symmetric, choice)), axis=(1, 2)) / scale
            maximum = max(maximum, float(error.max()))
            assert np.isfinite(actual).all() and error.max() < 2e-11, (kind, n, mode, choice, "upper poison")
            np.testing.assert_array_equal(actual, actual.transpose(0, 2, 1))
            assert np.linalg.eigvalsh(actual / scale[:, None, None]).min() > -2e-11
            np.testing.assert_array_equal(original.get().view(np.uint64), inputs.view(np.uint64))
    rows.append({"header": kind, "sizes": SIZES, "small_sizes": [1, 2, 3],
                 "legacy_comparison": "bitwise identical",
                 "upper_poison_projections": (len(SIZES)-1)*4*3*2, "maximum_scaled_error": maximum})
  return rows


def symbolic_check(output, projection):
  import yasps as y
  from yasps.codeGenerator import codeGenerator
  results = []
  for choice in (1, 2):
    directory = output / f"symbolic_{choice}"
    directory.mkdir()
    os.chdir(directory)
    scene = y.scene(f"evd_symbolic_{choice}")
    primitive = scene.addMesh("mesh").addPrimitive("matrices", 3)
    data = primitive.addAttribute("matrix", rows=6, cols=6)
    inputs = matrices(6)[[8, 9, 37]]
    data.updateValue(inputs)
    result = primitive.addAttribute("projected", computed_attribute=data.spd(choice))
    cache = directory / ".yasps_tmp"
    cache.mkdir(exist_ok=True)
    codeGenerator(result).generateCode()
    poisoned = [cache / f"compute_{result.fullNameWithHash}_spd_upper_v1.so"]
    poisoned += [cache / f"{item.attributeName}_spd_upper_v1.o" for item in result.deviceKernel.dependents + [result.deviceKernel]]
    for path in poisoned:
      path.write_bytes(b"stale Eigen cache: must never load or link")
    result.compute()
    actual = result.value.get().reshape(inputs.shape)
    np.testing.assert_allclose(actual, expected_projection(inputs, choice), rtol=2e-11, atol=2e-11)
    assert all(path.read_bytes() == b"stale Eigen cache: must never load or link" for path in poisoned)
    assert list(cache.glob(f"*{projection.PROJECTION_CACHE_TAG}*.so"))
    assert list(cache.glob(f"*{projection.PROJECTION_CACHE_TAG}*.o"))
    assert (cache / "allHeaders.cuh").read_text().startswith(projection.SMALL_EVD_HEADER)
    results.append({"choice": choice, "stale_files_ignored": len(poisoned)})
  return results


def dense_blocks(hessian):
  dense = np.zeros((hessian.rows, hessian.cols))
  offset = 0
  positions = hessian.block_positions.get().reshape(-1, 2)
  data = hessian.blocks_flattened.get()
  for group, count in enumerate(hessian.block_counts):
    rows, columns = hessian.block_dimensions[2*group:2*group+2]
    first = hessian.blocks_start_indices[group]
    blocks = data[first:first+count*rows*columns].reshape(count, rows, columns)
    for (row, column), block in zip(positions[offset:offset+count], blocks):
      dense[row:row+rows, column:column+columns] += block
      if row != column:
        dense[column:column+columns, row:row+rows] += block.T
    offset += count
  return dense


def hessian_checks(output, projection):
  import yasps as y
  from yasps.hessianAndGradientKernel import hessianAndGradientKernel
  results = []
  positions = np.array([[.3, -.7, 1.2], [-.6, .4, .8]])
  linear = np.array([[1., .2, -.1], [.3, 1.2, .4], [-.2, .1, .8]])
  weights = np.array([-1.5, .7, 1.3, -.8, 1.1, .5])
  for full in (False, True):
    for choice in (1, 2):
      directory = output / f"hessian_{'full' if full else 'staged'}_{choice}"
      directory.mkdir()
      os.chdir(directory)
      scene = y.scene(directory.name)
      mesh = scene.addMesh("mesh")
      nodes = mesh.addPrimitive("nodes", 2)
      x = nodes.addAttribute("x", rows=3, cols=1)
      x.updateValue(positions)
      mapped = nodes.addAttribute("mapped", computed_attribute=y.attribute.to_array(
        [sum(linear[i, j]*x[j] for j in range(3)) for i in range(3)], rows=3, cols=1))
      edge = mesh.addPrimitive("edge", 1)
      link = edge.addConnectivity("nodes", nodes, np.array([[0, 1]], dtype=np.uint32), 2)
      joined = edge.addAttribute("joined", through=link, source=mapped)
      total = sum(joined[i] for i in range(6))
      energy = edge.addAttribute("energy", computed_attribute=sum(.5*weights[i]*joined[i]*joined[i] for i in range(6)) + .01*total*total*total*total)
      hessian = y.differentiator().diff2([energy], [x], [x], projection_method=-1 if full else choice)
      if full:
        hessian.project_entire_hessian = [True]
        hessian.projection_methods = [choice]
      poisoned = []
      generate = hessianAndGradientKernel.generateKernel

      def seed_old_objects(kernel, *args, **kwargs):
        evaluation = kernel._hessianAndGradientKernel__evaluation
        for item in evaluation.kernels:
          path = directory / ".yasps_tmp" / f"{item.attributeName}_staged_v1_lto_0.o"
          path.write_bytes(b"stale staged Eigen object")
          poisoned.append(path)
        return generate(kernel, *args, **kwargs)

      with patch.object(hessianAndGradientKernel, "generateKernel", seed_old_objects):
        hessian.compute()
      z = (positions @ linear.T).ravel()
      inner_h = np.diag(weights) + .12*z.sum()**2*np.ones((6, 6))
      jacobian = np.kron(np.eye(2), linear)
      expected = expected_projection(jacobian.T @ inner_h @ jacobian, choice) if full else jacobian.T @ expected_projection(inner_h, choice) @ jacobian
      np.testing.assert_allclose(dense_blocks(hessian), expected, rtol=3e-11, atol=3e-12)
      np.testing.assert_allclose(hessian.gradient.value.get(), jacobian.T @ (weights*z+.04*z.sum()**3), rtol=3e-11, atol=3e-12)
      assert poisoned and all(path.read_bytes() == b"stale staged Eigen object" for path in poisoned)
      cache = directory / ".yasps_tmp"
      assert list(cache.glob(f"*{projection.PROJECTION_CACHE_TAG}*.o"))
      assert (cache / "allHeaders.cuh").read_text().startswith(projection.SMALL_EVD_HEADER)
      results.append({"full_projection": full, "choice": choice, "stale_objects_ignored": len(poisoned)})
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cpu-only", action="store_true")
  parser.add_argument("--output", type=Path)
  options = parser.parse_args()
  projection, helpers = source_checks()
  if options.cpu_only:
    print("PASS: packaged header, helper contracts and cache-version checks; no CUDA imported")
    return
  output = (options.output or Path(tempfile.mkdtemp(prefix="yasps-evd-integration-"))).resolve()
  output.mkdir(parents=True, exist_ok=True)
  previous = Path.cwd()
  report = {"header_sha256": HEADER_HASH, "cache_tag": projection.PROJECTION_CACHE_TAG, "status": "running"}
  try:
    report["helpers"] = helper_checks(helpers, output)
    report["symbolic"] = symbolic_check(output, projection)
    report["hessians"] = hessian_checks(output, projection)
    report["status"] = "passed"
  except Exception as error:
    report["status"], report["error"] = "failed", repr(error)
    raise
  finally:
    os.chdir(previous)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"PASS: {output / 'report.json'}")


if __name__ == "__main__":
  main()
