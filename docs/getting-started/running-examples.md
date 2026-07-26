# Running examples

The repository examples are executable Python programs rather than an
installed command-line suite. Install YASPS in editable mode first:

```bash
python -m pip install -e ./yasps
```

## CUDA examples

The examples are CUDA-oriented research programs.
Many assume that the current working directory is the example directory:

```bash
cd examples/one_bunny
python one_bunny.py
```

Some examples:

- execute immediately when imported;
- import a sibling `helpers.py`;
- use relative paths such as `../data/bunny.node`;
- require optional visualization packages such as PyVista;
- write to pre-existing `outputs/` or `meshes/` directories.

Read the top of an example before treating it as an importable module.

## Recommended validation progression

For a new backend or model change:

1. Build a scene with `--steps 0`.
2. Run one frame and one Newton iteration.
3. Run two frames so prior contacts become friction pairs.
4. Run past first impact.
5. Only then start a long output-producing simulation.

This isolates setup, solve, CCD, dynamic connectivity, and friction failures.
