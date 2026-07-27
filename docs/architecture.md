---
title: How YASPS executes
description: Follow a symbolic energy through differentiation, sparse indexing, CUDA code generation, assembly, and PCG.
permalink: /architecture/
---

<p class="eyebrow">Under the hood</p>

# How YASPS executes

YASPS keeps the energy symbolic until it can generate a kernel for the complete local computation. The high-level execution path is:

```text
scene hierarchy and symbolic attributes
        ↓
JOIN / UNION path discovery
        ↓
symbolic gradient and Hessian expressions
        ↓
sparse global block-index generation
        ↓
fused CUDA source generation and JIT compilation
        ↓
numeric gradient + Hessian block assembly
        ↓
block-diagonal preconditioned conjugate gradient
        ↓
one GPU solution segment per target
```

## 1. Symbolic graph

`attribute` nodes record an operator, shape, children, and correspondence. Python arithmetic builds this graph; it does not launch an elementwise array operation. Named attributes, JOINs, UNIONs, and requested outputs establish reusable code-generation boundaries.

The core implementation is in:

- `yasps/yasps/attribute/attribute.pyx` — graph nodes and public expression syntax;
- `yasps/yasps/attribute/attributeOperations.pyx` — operator-specific helpers;
- `yasps/yasps/operator/operator.pyx` — operator identifiers;
- `yasps/yasps/codeGenerator/` — expression-to-CUDA translation.

## 2. Differentiation paths

The differentiator first discovers every valid route from a scalar energy to the requested data attributes. A route can cross ordinary expression nodes, JOIN, or UNION.

- `yasps/yasps/differentiator/path.pyx` discovers roots and topology paths.
- `yasps/yasps/attribute/autodiff.pyx` implements local symbolic derivative rules.
- `yasps/yasps/differentiator/differentiator.pyx` composes local derivatives into global gradient and Hessian expressions.

The path layer is also where local energy targets are filtered and where variable-arity differentiation is currently rejected.

## 3. Sparse coordinate generation

The same paths determine which global target blocks a local energy instance touches. Generated index kernels emit block coordinates; coordinate compression merges duplicate coordinates and creates lookup arrays used during numerical assembly.

Relevant implementation:

- `yasps/yasps/kernel/Coordinate/gradientIndicesKernel.pyx`;
- `yasps/yasps/kernel/Coordinate/coordinateCompressionKernel.pyx`;
- `yasps/yasps/kernel/Coordinate/placementReorderKernel.pyx`;
- `yasps/yasps/differentiator/path.pyx`.

Static coordinates are cached after setup. Dynamic energy coordinates are recomputed when the active instance set changes.

## 4. CUDA code generation

Generated device expressions use Eigen matrices inside CUDA C++. Host-side generated C++ owns kernel launches and works with pointers obtained from PyCUDA allocations.

The generator and kernel wrappers live next to the subsystem that uses them:

- `yasps/yasps/codeGenerator/codeGenerator.pyx` emits attribute computation;
- `yasps/yasps/kernel/Compute/` handles ordinary `attribute.compute()`;
- `yasps/yasps/kernel/Hessian/` emits fused Hessian/gradient variants;
- `yasps/yasps/kernel/Coordinate/` emits index and compression kernels;
- `yasps/yasps/kernel/Solver/` emits diagonal inverse and PCG support.

Generated files are placed in `.yasps_tmp`; reusable fixed kernels are cached in `.yasps_constant`. The first evaluation of a new graph includes source generation and compilation, while later calls reuse the compiled artifacts and allocated buffers where possible.

## 5. Numerical assembly

A `hessian` stores symbolic term metadata plus block-sparse numerical buffers. On `compute()` it:

1. creates static sparse indices on first use;
2. refreshes dynamic indices when needed;
3. zeros reusable gradient, diagonal, and block buffers;
4. launches one fused assembly kernel per active energy term;
5. accumulates each local contribution through its sparse lookup.

The implementation is `yasps/yasps/matrixAndVector/matrix/hessian.pyx`. The `gradient` object in `yasps/yasps/matrixAndVector/vector/gradient.pyx` owns the global vector layout and per-target views.

## 6. Linear solve

`minimizer.computeNumericValue()` assembles the active Hessian and gradient, then computes inverse diagonal blocks for preconditioning. `solver.computeSolution()` invokes the generated PCG implementation on the GPU.

The system is `H Δx = g`; applying `x ← x − Δx` remains outside the library.

## Where to change a rule

| Change you want | Start here |
| --- | --- |
| Add or change public symbolic syntax | `attribute/attribute.pyx`, then `attributeOperations.pyx` |
| Change a local derivative rule | `attribute/autodiff.pyx` |
| Change JOIN/UNION derivative composition | `differentiator/differentiator.pyx` |
| Change lineage/path discovery | `differentiator/path.pyx` |
| Change global target offsets or sparse coordinate rules | `gradientIndicesKernel.pyx` and `path.pyx` |
| Change coordinate compression/block lookup | `coordinateCompressionKernel.pyx` |
| Change expression CUDA source | `codeGenerator/codeGenerator.pyx` and the relevant kernel wrapper |
| Change fused Hessian/gradient source | `kernel/Hessian/` |
| Change matrix assembly/buffer reuse | `matrixAndVector/matrix/hessian.pyx` |
| Change minimizer lifecycle or energy selection | `minimizer/minimizer.pyx` |
| Change PCG or preconditioning | `solver/solver.pyx` and `kernel/Solver/` |

Because YASPS is a compiled Cython package, changing a `.pyx` file requires rebuilding the extension before testing.

## Deliberate boundaries

The framework does not own collision candidate generation, CCD step-size computation, line search, Newton convergence policy, rendering, or simulation state integration. Those remain in examples and application code because they are policies around the generated objective rather than part of its symbolic representation.
