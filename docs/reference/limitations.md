# Limitations

This page records implementation behavior that is easy to miss.

## API and symbolic model

- Scene names are stored in a process-global registry and cannot be reused in
  one process.
- Names must be valid Python identifiers and must not collide with owner
  methods/properties.
- `attribute.reshape()` mutates metadata and returns `None`; use symbolic
  `resize()` in expressions.
- Python `attribute_a == attribute_b` is structural/hash equality, not a
  numerical node. Use `.eq()`.
- `*` is scalar scaling or matrix multiplication, never Hadamard multiply.
- `SUM`/`AVERAGE` variable-arity gathers are computational operators but not
  a fully supported general energy differentiation path.
- Raw data/constants can technically be created on primitive unions, but the
  portable pattern is a child-backed named union attribute.
- `scene.energies` is not populated by the minimizer registration path.
- `gradient_only=True` and the first-derivative-only helper path are not
  complete user-facing solver modes.

## Input validation

`updateValue()` flattens input but does not consistently reject every scalar
count mismatch. Validate application arrays before upload.

Connectivity code assumes indices are in range. Invalid indices can become a
GPU memory/indexing failure rather than a friendly model error.

## CUDA

- Generated kernels assume a working `nvcc`, CUDA runtime/driver linker, and
  Eigen headers at runtime.
- Several commands currently hard-code `sm_89`; other architectures may need
  build-command changes.
- Generated files and shared libraries are stored under `.yasps_tmp` and
  related cache paths.
- The example CCD helper headers are bundled from the MPL-2.0 GIPC source.

## Examples

Most original examples:

- are CUDA-oriented;
- execute at import time;
- depend on their current working directory;
- may require PyVista or other undeclared optional packages;
- may assume output directories exist.
