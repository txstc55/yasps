# Dropping in a container on Metal

> This page is intentionally excluded from the published documentation while
> Metal validation is in progress.

`dropping_in_container_metal.py` is a thin launcher. It sets
`YASPS_BACKEND=metal`, creates the expected output directories, changes into
the example directory, and executes the original
`dropping_in_container.py`. The material model, time step, collision
parameters, solver settings, and line search therefore have one source of
truth.

## Run

From the repository root:

```bash
PYTHONPATH="$PWD/yasps:$PWD/examples/ccd" \
  .venv/bin/python \
  examples/dropping_in_container/dropping_in_container_metal.py
```

The original driver accepts:

```bash
--num-bunnies N
```

For bounded validation, the shared source also honors:

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `YASPS_EXAMPLE_FRAMES` | Frames to execute | `500` |
| `YASPS_EXAMPLE_SHOW` | Open/update interactive PyVista | `1` |
| `YASPS_EXAMPLE_SAVE` | Save screenshots/OBJ outputs | `1` |

A compute-only smoke test is:

```bash
PYTHONPATH="$PWD/yasps:$PWD/examples/ccd" \
YASPS_EXAMPLE_FRAMES=3 \
YASPS_EXAMPLE_SHOW=0 \
YASPS_EXAMPLE_SAVE=0 \
  .venv/bin/python \
  examples/dropping_in_container/dropping_in_container_metal.py
```

These controls do not change scene physics or solver parameters.

## Scene structure

The driver creates deformable bunny vertices and tetrahedra, fixed container
geometry, and a collision mesh with a primitive union over both vertex
collections. The union exposes matching `position` and historical attributes
without erasing each child's derivative path.

Tetrahedra JOIN four current and four rest positions. The stable
Neo-Hookean energy uses generated determinant and inverse operations and is
registered alongside inertia/gravity.

## Dynamic contact

The collision mesh owns dynamic PP, PE, PT, and EE primitives. After discrete
classification, the driver:

1. updates each dynamic primitive's active count;
2. writes its connectivity;
3. evaluates the existing barrier graph; and
4. asks the dynamic Hessian path to regenerate active indices and values.

The expression graph is not rebuilt for every pair set.

At the beginning of a frame, the prior accepted contact set becomes the
lagged friction set. Closest coordinates, tangent bases, and normal-force
magnitudes are named computed attributes so they can be reused in the
friction expressions and generated modules.

## Newton and CCD loop

Each nonlinear iteration:

1. evaluates the current energy;
2. generates/assembles the active Hessian and gradient;
3. solves \(H\Delta x=g\);
4. forms the proposed device-resident displacement;
5. runs swept face/edge CCD on Metal;
6. limits the step with generated time-of-impact kernels;
7. updates discrete contacts at the trial state; and
8. accepts a descending energy or backtracks.

If float32 line search cannot find a descending representable state, the
driver restores the last accepted state. Frame velocities are updated only
after the nonlinear loop.

## Rendering

Default runs preserve the original interactive PyVista behavior. For
performance measurements, set both display and saving to zero so VTK does not
compete with compute kernels on the same interactive Apple GPU.

An off-screen export can be validated with:

```bash
PYVISTA_OFF_SCREEN=true \
YASPS_EXAMPLE_FRAMES=1 \
YASPS_EXAMPLE_SHOW=0 \
YASPS_EXAMPLE_SAVE=1 \
  .venv/bin/python \
  examples/dropping_in_container/dropping_in_container_metal.py
```

Generated JPG/OBJ outputs are ignored by Git and should be removed after
validation.
