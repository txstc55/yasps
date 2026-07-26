# Installation

YASPS requires Python 3.10 or newer and an NVIDIA CUDA GPU.

## Requirements

Requirements:

- a supported NVIDIA GPU and driver;
- CUDA Toolkit, including `nvcc`;
- PyCUDA;
- Eigen headers for generated CUDA/Eigen kernels;
- Python 3.10 or newer.

Install the package from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./yasps
```

YASPS data and generated CUDA kernels use float64.

!!! note

    Several CUDA kernels are generated and compiled on first use. A successful
    Python package build does not by itself prove that `nvcc`, Eigen, and the
    CUDA runtime linker are configured.

## Editable builds after Cython changes

Changes to `.pyx` files require rebuilding the extension modules:

```bash
python -m pip install -e ./yasps --no-build-isolation
```

## Documentation dependencies

The documentation site has an intentionally separate dependency:

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

`mkdocs serve` watches the Markdown files and reloads the local site.
