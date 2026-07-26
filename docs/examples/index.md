# Example map

The examples are the most complete source of YASPS usage patterns. They are
research programs rather than a uniform test suite: most configure a scene at
module scope, several depend on their current working directory, and some
expect optional visualization or mesh-processing packages.

Start with the smaller examples and use the large collision programs as
references for individual techniques.

## Suggested reading order

| Example | Main ideas |
| --- | --- |
| `one_bunny/one_bunny.py` | Tetrahedral primitives, gathered positions, stable Neo-Hookean elasticity, inertia, and a Newton update |
| `pendulum_bdf2/pendulum_bdf2.py` | Time integration and updating state between steps |
| `one_bunny_partial_abd/one_bunny_partial_abd.py` | Multiple kinds of degrees of freedom and affine-body coordinates |
| `two_bunnies_abd_soft/two_bunnies_abd_soft.py` | Primitive unions combining affine and deformable geometry |
| `smoothing/smoothing.py` | A compact non-dynamics computation expressed through attributes |
| `repulsive/repulsive.py` | Neighborhood-like interactions and repulsive energies |
| `many_bunnies_one_cloth_with_cage_energy/` | Mixed element types, cages, cloth, affine bodies, unions, and many energy families |
| `dropping_in_container/dropping_in_container.py` | Full elasticity, inertia, IPC contact, friction, CCD, Newton, line search, and velocity updates |

## Deformable solids

`one_bunny` is the clearest elasticity example. It uses:

- a vertex primitive with current, rest, previous, and velocity attributes;
- a tetrahedron primitive connected to four vertices per element;
- `JOIN` attributes that gather four positions into every tetrahedron;
- a scalar stable Neo-Hookean energy per tetrahedron;
- an inertia energy per vertex;
- `scene.addMinimizeTarget([position])`; and
- a Newton solve followed by `position - delta`.

The `two_bunnies` and dropping examples extend the same structure with
collision geometry.

## Affine bodies and mixed coordinates

The ABD examples use a 3×3 affine matrix and a translation as degrees of
freedom. A vertex position is computed symbolically from those attributes and
a rest-space coordinate:

```python
position = affine_matrix * rest_position + translation
vertices.addAttribute("position", computed_attribute=position)
```

Because the computed position retains lineage to both targets, energies written
using the position differentiate back to the affine matrix and translation.
The mixed ABD/soft examples demonstrate that a primitive union may expose a
common `"position"` attribute even when each child computes it from different
degrees of freedom.

## Cloth, cages, and embedding

The `many_bunnies_one_cloth*` examples combine:

- triangle membrane energies;
- edge or four-vertex bending stencils;
- tetrahedral and hexahedral solid energies;
- cage weights that interpolate embedded vertices; and
- affine and free vertex coordinates in one solve.

They are useful references for longer `JOIN` chains and for registering several
minimization targets in a known order.

## Collision and friction

The large contact examples create four dynamic primitive types:

| Type | Connectivity width | Meaning |
| --- | ---: | --- |
| `pp` | 2 | point–point |
| `pe` | 3 | point–edge |
| `pt` | 4 | point–triangle |
| `ee` | 4 | edge–edge |

Collision detection supplies the connectivity arrays at runtime. The program
updates each primitive's active instance count, uploads the new indices, and
then evaluates the already-defined barrier or friction expression. This is the
central benefit of dynamic primitives: the symbolic energy graph is compiled
once even though the interaction set changes every frame or line-search trial.

See [Dynamic collision terms](../guides/dynamic-collision-terms.md) for the
construction pattern.

## Running an example safely

Before running an older example:

1. inspect its imports and path literals;
2. launch it from its own directory when it uses relative paths;
3. create any expected output directory;
4. disable or install its visualization dependency; and
5. begin with a very small step count.

The CUDA examples are research artifacts and have not all been converted into
portable command-line programs.
