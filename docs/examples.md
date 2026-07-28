---
title: Examples map
description: A guided map of the simulation, collision, deformation, and benchmark examples in the repository.
permalink: /examples/
---

<p class="eyebrow">Learn from working programs</p>

# Examples map

The examples are research programs rather than a uniform command-line suite. Run each entry point from its own directory so relative mesh, cache, and output paths resolve correctly.

```bash
cd examples/one_bunny
python one_bunny.py
```

Most simulations need the optional geometry and visualization packages imported by their local `helpers.py`, in addition to YASPS's core CUDA dependencies.

> For a guided reading of hierarchy construction, two parameterizations, JOIN, UNION, dynamic contact, separated Hessian/Jacobian assembly, target layout, CCD, and line search, follow the [mixed bodies with separated assembly tutorial]({{ '/tutorials/mixed-separation/' | relative_url }}?v={{ site.time | date: '%s' }}).

## Recommended reading order

1. [one_bunny](https://github.com/txstc55/yasps/tree/main/examples/one_bunny) — the clearest complete soft-body IPC loop.
2. [one_bunny_partial_abd](https://github.com/txstc55/yasps/tree/main/examples/one_bunny_partial_abd) — several target parameterizations joined by a primitive union.
3. [dropping_in_container](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container) — a larger contact scene with friction and a container.
4. [dropping_in_container_mixed](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container_mixed) — soft and affine bodies in one collision system.
5. [pendulum_bdf2](https://github.com/txstc55/yasps/tree/main/examples/pendulum_bdf2) — multi-object time integration, constraints, and collision.

When reading an example, start where its scene, meshes, primitives, and attributes are declared. Then follow `addEnergy` and `addMinimizeTarget`. The loop after those calls shows which responsibilities remain outside YASPS.

## Deformation and contact

| Example | Main ideas |
| --- | --- |
| [one_bunny](https://github.com/txstc55/yasps/tree/main/examples/one_bunny) | Stable Neo-Hookean tetrahedra, inertia, fixed/free vertex UNION, dynamic PP/PE/PT/EE contact |
| [two_bunnies](https://github.com/txstc55/yasps/tree/main/examples/two_bunnies) | Multiple deformable objects and self/inter-object contact |
| [dropping_in_container](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container) | Repeated bodies, container contact, friction, output and analysis helpers |
| [brazil_nuts](https://github.com/txstc55/yasps/tree/main/examples/brazil_nuts) | Dense mixed-body contact with friction and several target groups |
| [bunny_wrap](https://github.com/txstc55/yasps/tree/main/examples/bunny_wrap) | Cloth/surface energies wrapping around another object |
| [mattwist](https://github.com/txstc55/yasps/tree/main/examples/mattwist) | Cloth inertia, bending, Baraff–Witkin energy, twisting constraint, and IPC |
| [smoothing](https://github.com/txstc55/yasps/tree/main/examples/smoothing) | Surface smoothing and radius control combined with contact |

## Affine and mixed parameterizations

| Example | Main ideas |
| --- | --- |
| [two_bunnies_abd](https://github.com/txstc55/yasps/tree/main/examples/two_bunnies_abd) | Affine-body translations and matrices as minimization targets |
| [one_bunny_partial_abd](https://github.com/txstc55/yasps/tree/main/examples/one_bunny_partial_abd) | Fixed, soft, and affine regions in one bunny |
| [one_bunny_partial_abd_separate_jacobian](https://github.com/txstc55/yasps/tree/main/examples/one_bunny_partial_abd_separate_jacobian) | The same model using separate Hessian/Jacobian code generation |
| [two_bunnies_abd_soft](https://github.com/txstc55/yasps/tree/main/examples/two_bunnies_abd_soft) | Soft and affine bodies combined through UNION |
| [dropping_in_container_mixed](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container_mixed) | Mixed soft/affine population in a container |
| [dropping_in_container_mixed_separation](https://github.com/txstc55/yasps/tree/main/examples/dropping_in_container_mixed_separation) | Mixed container scene with separated derivative stages |
| [two_coins](https://github.com/txstc55/yasps/tree/main/examples/two_coins) | Affine and elastic terms on a compact two-object setup |

## Cages, cloth, and combined scenes

| Example | Main ideas |
| --- | --- |
| [cage](https://github.com/txstc55/yasps/tree/main/examples/cage) | Cage degrees of freedom, interpolated bunny vertices, floor contact |
| [many_bunnies_one_cloth](https://github.com/txstc55/yasps/tree/main/examples/many_bunnies_one_cloth) | Many bodies interacting with a cloth |
| [many_bunnies_one_cloth_with_cage](https://github.com/txstc55/yasps/tree/main/examples/many_bunnies_one_cloth_with_cage) | Soft, affine, cloth, and tetrahedral cage targets |
| [many_bunnies_one_cloth_with_cage_energy](https://github.com/txstc55/yasps/tree/main/examples/many_bunnies_one_cloth_with_cage_energy) | Hexahedral cage energy split into generated terms |
| [one_bunny_many_cloths](https://github.com/txstc55/yasps/tree/main/examples/one_bunny_many_cloths) | One body interacting with several cloths |
| [one_bunny_many_cloths_optimized](https://github.com/txstc55/yasps/tree/main/examples/one_bunny_many_cloths_optimized) | A performance-oriented version for comparison |
| [teaser](https://github.com/txstc55/yasps/tree/main/examples/teaser) | Large showcase combining affine, soft, cage, and collision systems |

## Specialized objectives

| Example | Main ideas |
| --- | --- |
| [pendulum_bdf2](https://github.com/txstc55/yasps/tree/main/examples/pendulum_bdf2) | BDF2 integration, string and volumetric bodies, constraints, mixed collision |
| [repulsive](https://github.com/txstc55/yasps/tree/main/examples/repulsive) | Curve energies, surface weight processing, and collision |
| [repulsive_in_bunny](https://github.com/txstc55/yasps/tree/main/examples/repulsive_in_bunny) | Optimize a loop inside a fixed bunny |
| [repulsive_on_bunny](https://github.com/txstc55/yasps/tree/main/examples/repulsive_on_bunny) | Repulsive loop objective on a bunny surface |

`two_bunnies_soft_surface` currently contains supporting helper code rather than a standalone top-level Python entry point.

## Utilities and performance studies

- [ccd](https://github.com/txstc55/yasps/tree/main/examples/ccd) contains a standalone continuous-collision-detection experiment.
- [plotting](https://github.com/txstc55/yasps/tree/main/examples/plotting) contains scripts for compile time, code size, sparse-index, Hessian, and projection figures.
- `examples/data` contains shared geometry and other input data.
- `examples/backup` contains older experiments; treat current top-level examples as the maintained syntax reference.

## What to copy—and what to reconsider

The examples are authoritative for composition patterns: named expressions, JOIN, UNION, dynamic contact primitives, projection flags, and the sign of the returned direction. Constants such as timestep, tolerance, material values, CUDA architecture, iteration count, file paths, and stopping rules are experiment-specific and should not be copied blindly.
