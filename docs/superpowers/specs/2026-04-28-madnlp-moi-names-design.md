# MadNLPMOI: support for `MOI.VariableName` and `MOI.ConstraintName`

**Date:** 2026-04-28
**Author:** Guillaume Marques
**Status:** Approved — ready for implementation plan

## Context

The current `MadNLPMOI` extension does not implement `MOI.VariableName` or
`MOI.ConstraintName`. When MadNLP is used through JuMP's default
`Model(MadNLP.Optimizer)` path, names are kept by the surrounding
`MOI.Utilities.CachingOptimizer` — but they never reach the MadNLP backend.
A user calling `unsafe_backend(model)` directly, or constructing the optimizer
in `direct_model` mode, has no way to recover the names of variables and
constraints from the `MadNLP.Optimizer` or from the underlying NLP model.

This spec defines a focused extension to bridge that gap: store names at the
MOI layer, propagate them to the `MOIModel` (the `AbstractNLPModel` wrapper),
so application code can read them after `optimize!` via the solver's `nlp`
field.

## Goals

- `MOI.set/get` for `MOI.VariableName` and `MOI.ConstraintName` work on
  `MadNLP.Optimizer` (i.e. the `MadNLPMOI.Optimizer` extension type).
- `MOI.supports` reports `true` for these attributes (with the standard MOI
  exception that `ConstraintName` of a `CI{VariableIndex,S}` is unsupported).
- Inverse lookup `MOI.get(opt, MOI.VariableIndex, "name")` and
  `MOI.get(opt, MOI.ConstraintIndex, "name")` work.
- After `optimize!`, the underlying `MOIModel` (accessible as `opt.nlp`)
  carries `var_names::Vector{String}` and `con_names::Vector{String}` aligned
  with the variable and constraint orderings used by MadNLP's evaluator.
- Helpers `MadNLPMOI.get_variable_names(opt)` and
  `MadNLPMOI.get_constraint_names(opt)` return the propagated vectors,
  transparently descending through `MadNLP.SparseWrapperModel` if present.

## Non-goals

- The MadNLP IPM solver, printer, and logger are not modified. Names are not
  used in EXIT messages, inertia warnings, or any other internal output.
- The base `MadNLP` module does not gain new exports; the API stays inside the
  `MadNLPMOI` extension.
- No GPU-specific path: names live as `Vector{String}` on the host. Acceptable
  because they participate in no numerical computation.
- No automatic fallback name generation (e.g. `"x[1]"`). Missing names are
  represented by the empty string `""`.

## Design

### 1. Storage on `Optimizer`

Add to the `Optimizer` struct in `ext/MadNLPMOI/MOI_wrapper.jl`:

```julia
var_names::Dict{MOI.VariableIndex, String}
con_names::Dict{MOI.ConstraintIndex, String}
```

Both are initialized empty in the `Optimizer()` constructor and emptied in
`MOI.empty!(model::Optimizer)`. The `MOI.delete` methods for variables and
constraints must also remove the corresponding entries.

#### MOI methods

```julia
MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true
MOI.supports(::Optimizer, ::MOI.ConstraintName, ::Type{<:MOI.ConstraintIndex}) = true
# Standard MOI exception: variable-bound constraints don't get names.
MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintName,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex, S}},
) where {S} = false

MOI.set(opt::Optimizer, ::MOI.VariableName, vi::MOI.VariableIndex, name::String)
MOI.get(opt::Optimizer, ::MOI.VariableName, vi::MOI.VariableIndex)::String

MOI.set(opt::Optimizer, ::MOI.ConstraintName, ci::MOI.ConstraintIndex, name::String)
MOI.get(opt::Optimizer, ::MOI.ConstraintName, ci::MOI.ConstraintIndex)::String

# Inverse lookup
MOI.get(opt::Optimizer, ::Type{MOI.VariableIndex}, name::String)
MOI.get(opt::Optimizer, ::Type{C}, name::String) where {C<:MOI.ConstraintIndex}
```

Missing entries return `""` on `MOI.get` for `VariableName`/`ConstraintName`.
Inverse lookup returns `nothing` if the name is not found.

#### Duplicate name policy

`MOI.set` rejects setting a non-empty name that already exists for another
index (consistent with `MOI.Utilities.Model`). The error type is
`MOI.SettingAttributeNotAllowed` with a clear message. Setting `""` always
succeeds (clears the entry).

### 2. Propagation to `MOIModel`

Extend the `MOIModel` struct (same file) with two new fields:

```julia
struct MOIModel{T} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    model::Optimizer
    counters::NLPModels.Counters
    var_names::Vector{String}   # length == meta.nvar
    con_names::Vector{String}   # length == meta.ncon
end
```

Built inside `_setup_nlp` (around current line 1437), right after `nvar` and
`ncon` are known.

#### `var_names` construction

```julia
var_names = String[
    get(model.var_names, vi, "") for vi in model.list_of_variable_indices
]
```

`var_names` has length `meta.nvar` of the **untransformed** model — i.e. the
nvar before any `MakeParameter` rewrite of fixed variables. Documented as the
choice in the docstring of `MOIModel`. Users who want names of the active
(free) variables filter with `solver.cb.fixed_handler.free`.

#### `con_names` construction

The order must match the layout used by `eval_constraint`,
`jacobian_structure`, etc. Inspecting the existing wrapper:

1. **`qp_data`** (linear/quadratic) — rows `1 : length(qp_data)`, ordered by
   the internal `QPBlockData` ordering. Implemented via a helper
   `_qp_constraint_indices(qp_data)` that returns the `MOI.ConstraintIndex`
   list in row order. The `QPBlockData` already tracks added CIs; we add a
   small accessor in `ext/MadNLPMOI/MOI_utils.jl` if not present.
2. **`vector_nonlinear_oracle_constraints`** — each oracle adds
   `s.set.output_dimension` rows. Each row is labeled with the same name (the
   `MOI.ConstraintName` of the parent oracle CI). Indexing `[k]` is **not**
   appended; if a user wants per-row labels, they can post-process. Rationale:
   the CI is a vector constraint with one user-set name.
3. **`nlp_data`** (NL block from `model.nlp_model`) — ordered by the
   `MOI.Nonlinear.ConstraintIndex` ordering. Each `Nonlinear.ConstraintIndex`
   is mapped back to its MOI `ConstraintIndex` via the existing bookkeeping in
   `model.nlp_model` / `model.mult_g_nlp`. If the mapping is not available
   directly, store an auxiliary `Vector{MOI.ConstraintIndex}` during NL
   constraint addition.

Each row's name is `get(model.con_names, ci, "")`.

### 3. Helpers and `SparseWrapperModel`

When MadNLP runs with a non-default `array_type` (e.g. CUDA), `_setup_nlp`
wraps `MOIModel` in `MadNLP.SparseWrapperModel`. To present a stable API:

```julia
get_variable_names(opt::Optimizer)   = _names_of(opt.nlp).var_names
get_constraint_names(opt::Optimizer) = _names_of(opt.nlp).con_names

_names_of(nlp::MOIModel)                  = nlp
_names_of(nlp::MadNLP.SparseWrapperModel) = _names_of(nlp.inner)
_names_of(::Nothing)                      = (var_names = String[], con_names = String[])
```

Both helpers are exported from the `MadNLPMOI` module so they can be reached
as `MadNLP.MadNLPMOI.get_variable_names(opt)` (or directly through the
extension's namespace once loaded).

### 4. Fixed-variable handling (explicitly addressed)

When `fixed_variable_treatment = MakeParameter` and the callback is a
`SparseCallback`, MadNLP removes fixed variables and the **callback** sees a
reduced variable space. The `MOIModel.meta.nvar` returned by `_setup_nlp` is
still the original count; the reduction happens later inside the IPM
callback. So `var_names` of length `meta.nvar` is well-defined and matches
`model.list_of_variable_indices`.

This means: **`solver.cb` does NOT carry names** — only the original
`MOIModel` does. Documented limitation.

### 5. Testing

Add `test/MOI_names_test.jl`, included from `test/runtests.jl`. Naming
follows `~/.claude/CLAUDE.md` (Nablarise conventions):

- Testset prefix `"[moi_names] ..."`.
- Functions `test_moi_names_<behavior>()`, lowercase with underscores.
- Each function builds its own model — no shared mutable state.
- No floating-point equality (`≈` only).
- Each new `.jl` file starts with the Nablarise license header.

Cases:

1. `test_moi_names_supports_attributes` — `MOI.supports` returns expected
   booleans for `VariableName`, `ConstraintName`, and the variable-bound CI
   exception.
2. `test_moi_names_variable_set_get` — set then get a variable name.
3. `test_moi_names_constraint_set_get_affine` — set/get on an affine CI.
4. `test_moi_names_constraint_set_get_nl` — set/get on an NL block CI.
5. `test_moi_names_inverse_lookup_variable` —
   `MOI.get(opt, MOI.VariableIndex, "x")` returns the right index;
   missing name returns `nothing`.
6. `test_moi_names_inverse_lookup_constraint` — same for constraints.
7. `test_moi_names_duplicate_rejected` — setting a duplicate name throws.
8. `test_moi_names_empty_default` — variable/constraint without an explicit
   name yields `""` from `MOI.get` and from the propagated vectors.
9. `test_moi_names_empty_clears` — after `MOI.empty!(opt)`, both dicts are
   empty.
10. `test_moi_names_propagation_to_nlpmodel` — after `_setup_nlp` (or
    `optimize!` with `max_iter=1`), `opt.nlp.var_names` and `opt.nlp.con_names`
    have the expected length and content.
11. `test_moi_names_jump_roundtrip` — full JuMP path with `set_name`,
    `max_iter = 1`, then `MadNLPMOI.get_variable_names(unsafe_backend(model))`
    matches the names set on the JuMP model.
12. `test_moi_names_delete_clears_entry` — deleting a variable or constraint
    removes its name from the dict.

The existing `test_MOI_Test` (which calls `MOI.Test.runtests`) is left
untouched. After implementation, re-run it: some `Name`-related tests may now
pass without exclusion. If new tests in the MOI test suite become applicable,
remove their exclusions accordingly.

## Risks and open points

- **`QPBlockData` introspection**: the helper `_qp_constraint_indices` must
  return CIs in the exact row order the rest of the code uses. Verify by
  cross-checking with the affine/quadratic Jacobian-row layout in
  `MOI_utils.jl`. If the existing code does not expose this, the cleanest fix
  is a small accessor inside `MOI_utils.jl` that mirrors the internal
  `Vector{ConstraintIndex}` of `QPBlockData`.
- **NL block CI mapping**: confirm `model.nlp_model` exposes constraint
  indices in iteration order. If not, maintain an auxiliary
  `Vector{MOI.ConstraintIndex}` populated when constraints are added through
  `MOI.add_constraint(::Optimizer, ::MOI.ScalarNonlinearFunction, ...)`.
- **`SparseWrapperModel` stability**: the `inner` field name is part of the
  internal MadNLP API. If it changes, `_names_of` breaks. Acceptable since the
  helper lives in the same repo as `SparseWrapperModel`.

## 6. KKT augmented and Hessian row/column labels

This is the primary practical motivation. After the basic name propagation, we
add helpers that return labels aligned with the augmented KKT and Hessian
matrices in MadNLP's internal order — so the user can call them on a
`MadNLPSolver` and inspect coefficient magnitudes by name.

### Augmented KKT layout (reference)

For `SparseKKTSystem` and `DenseKKTSystem` (reduced form), the augmented
matrix has size `n_tot + m` where `n_tot = cb.nvar + length(cb.ind_ineq)`:

| Row range                         | Meaning                                         |
|-----------------------------------|-------------------------------------------------|
| `1 .. cb.nvar`                    | Primal variables (free, in `cb` order)          |
| `cb.nvar+1 .. n_tot`              | Slack variables (one per inequality)            |
| `n_tot+1 .. n_tot+cb.ncon`        | Constraint multipliers (in `cb` order)          |

For `SparseUnreducedKKTSystem` (size `n_tot + m + nlb + nub`), append:

| Row range                                | Meaning                                     |
|------------------------------------------|---------------------------------------------|
| `n_tot+m+1 .. n_tot+m+nlb`               | Lower-bound multipliers (zL), `cb.ind_lb`   |
| `n_tot+m+nlb+1 .. n_tot+m+nlb+nub`       | Upper-bound multipliers (zU), `cb.ind_ub`   |

The Hessian of the Lagrangian is `n_tot × n_tot` (free variables + slacks).
Slack rows/cols of the Hessian are zero by construction.

### Mapping from `cb` indices back to original variable/constraint names

`SparseCallback` and `DenseCallback` carry `fixed_handler::AbstractFixedVariableTreatment`:

- `NoFixedVariables`: `cb.nvar == nlp.meta.nvar`, mapping is identity.
- `MakeParameter`: `cb.fixed_handler.free::Vector{Int}` lists original indices
  of free variables; the i-th free variable in `cb` corresponds to original
  variable `cb.fixed_handler.free[i]`.
- `RelaxBound`: same as `NoFixedVariables` for indexing purposes.

Constraint indices are not reordered by the callback (no fixed-constraint
treatment), so `cb` constraint k maps directly to MOIModel constraint k.

### Label conventions

- Primal variable label: `nlp.var_names[orig_idx]`, falling back to
  `"x[<orig_idx>]"` if the stored name is empty.
- Constraint label: `nlp.con_names[k]`, falling back to `"c[<k>]"`.
- Slack label: `"slack[<constraint label>]"` for the slack on the k-th
  inequality constraint, where `<constraint label>` is the constraint label
  for `cb.ind_ineq[k]`.
- Lower bound multiplier label: `"zL[<primal-or-slack label>]"` for index
  `cb.ind_lb[k]` in the `n_tot` space (so it can be a slack name when the
  lower bound is on a slack).
- Upper bound multiplier label: `"zU[<...>]"`, same scheme with `cb.ind_ub`.
- Constraint multiplier label: `"λ[<constraint label>]"`.

Fallback logic prevents empty strings in user-facing output. The fallback
format is documented in the helper docstrings.

### Public API

Added in the `MadNLPMOI` extension module:

```julia
"""
    kkt_row_labels(solver::MadNLP.MadNLPSolver) -> Vector{String}

Return labels for each row of the augmented KKT matrix
`solver.kkt.aug_com` (or `aug_raw`), in MadNLP's internal order.

Length matches `size(solver.kkt.aug_com, 1)`. Works for both reduced
(`SparseKKTSystem`, `DenseKKTSystem`) and unreduced
(`SparseUnreducedKKTSystem`) systems.
"""
kkt_row_labels(solver::MadNLP.MadNLPSolver)::Vector{String}

"""
    kkt_col_labels(solver::MadNLP.MadNLPSolver) -> Vector{String}

Same as `kkt_row_labels` (the augmented KKT is symmetric).
"""
kkt_col_labels(solver::MadNLP.MadNLPSolver)::Vector{String}

"""
    hessian_labels(solver::MadNLP.MadNLPSolver) -> Vector{String}

Return labels for each row/column of the Lagrangian Hessian,
of length `cb.nvar + length(cb.ind_ineq)` (primals + slacks).
"""
hessian_labels(solver::MadNLP.MadNLPSolver)::Vector{String}
```

These helpers do not modify `MadNLPSolver` or any KKT struct — they only
read existing fields (`solver.cb`, `solver.kkt`, and the `MOIModel` reached
through `solver.cb.nlp` or its inner if wrapped).

### Dispatch on KKT system type

Internally, `kkt_row_labels` dispatches on the concrete KKT system:

```julia
kkt_row_labels(solver::MadNLPSolver) =
    _kkt_row_labels(solver.kkt, solver.cb, _nlp_with_names(solver.cb.nlp))

_kkt_row_labels(kkt::AbstractReducedKKTSystem, cb, nlp) = ...
_kkt_row_labels(kkt::AbstractUnreducedKKTSystem, cb, nlp) = ...

_nlp_with_names(nlp::MOIModel) = nlp
_nlp_with_names(nlp::MadNLP.SparseWrapperModel) = _nlp_with_names(nlp.inner)
```

If the underlying NLP is not a `MadNLPMOI.MOIModel` (e.g. a user-provided
custom NLPModel), all labels fall back to the `"x[i]"` / `"c[k]"` form. This
keeps the helpers usable in all cases, just without semantic names.

## 7. Validation examples (NLP with distinctive coefficients)

Six small NLP problems serve as integration tests. Each uses **prime
numbers, very different magnitudes, or unique nonlinear functions** so each
nonzero in the Hessian and Jacobian can be matched by inspection to a single
contributing term — making it easy to verify that the labels map to the
correct row/column.

Each example is solved with `max_iter = 1` (we only need the first KKT
matrix to validate labels and structure; we do not need the optimum). The
test reads `solver.kkt.aug_com` and `solver.kkt.hess_com`, asserts the
shape, and checks selected entries against the expected derivative value at
the starting point.

### Example 1 — Pure QP, very different magnitudes

```
min  1000 x^2 + 0.01 y^2 + 7 x y
s.t. 5 x + 0.1 y >= 1     (c_lin1)
     x - 100 y <= 50      (c_lin2)
```

- Hessian (n=2): `[2000 7; 7 0.02]`. Each entry is unique.
- Jacobian rows: `[5  0.1]` and `[1  -100]`.
- Validates: variable label order, basic linear constraint label propagation.

### Example 2 — Nonlinear with primes and exponentials

```
min  exp(3 x) + log(1 + y^2)
s.t. x * y - 13 == 0       (c_eq)
```

- Hessian at x0=(0, 1): xx = `9*exp(0) = 9`, yy depends on y, off-diag from
  `xy` term in the Lagrangian (constraint contributes `λ` to `H_{xy}`).
- Validates: equality-constraint label, multiplier label `λ[c_eq]`.

### Example 3 — Mixed linear + NL, three variables

```
min  x^2/2 + 100 y^2 + sin(z)
s.t. x + y + z <= 10              (c_sum, linear)
     x * y >= 0.5                 (c_bilin, NL inequality)
     42 z^3 - 7 == 0              (c_cubic, NL equality)
```

- 3 vars, 3 constraints (1 lin + 1 NL ineq + 1 NL eq).
- 1 inequality → 1 slack. Augmented KKT size = `3 + 1 + 3 = 7` (reduced).
- Hessian features: 1 (xx), 200 (yy), `-sin(z)` (zz). Off-diag from
  bilinear `x*y` constraint Lagrangian contribution (`λ_bilin`).
- Validates: slack label `slack[c_bilin]`, ordering primal → slack →
  multipliers, mixing of linear and NL constraints in `con_names`.

### Example 4 — Bound multipliers (unreduced KKT)

```
min  (x - 3)^2 + (y + 2)^2
s.t. 0 <= x <= 100
     -50 <= y <= 50
```

- 2 vars, 0 explicit constraints → m = 0. Both vars have lower AND upper
  bounds → `nlb = nub = 2`.
- Solved with `kkt_system = SparseUnreducedKKTSystem` to expose `zL`/`zU`.
- Augmented KKT size = `2 + 0 + 0 + 2 + 2 = 6`.
- Validates: `zL[x]`, `zL[y]`, `zU[x]`, `zU[y]` labels.

### Example 5 — Fixed variable

```
min  (x - 1)^2 + (y - 2)^2 + (z - 3)^2
s.t. x == 5            (fixed via equal lower/upper bound on x)
     x + 2 y + 3 z >= 6
```

- With `fixed_variable_treatment = MakeParameter` and a `SparseCallback`,
  the fixed variable `x` is removed → `cb.nvar = 2`.
- `cb.fixed_handler.free` should be `[2, 3]` (original indices of `y`, `z`).
- Validates: variable labels `kkt_row_labels(solver)[1] == name(y)` and
  `[2] == name(z)` — i.e. the fixed `x` is *skipped*, not just renamed.

### Example 6 — Equality with distinctive prime Jacobian

```
min  w^2 + x^2 + y^2 + z^2
s.t. 13 w + 17 x + 19 y + 23 z == 100      (c_primes)
```

- 4 vars, 1 eq constraint. Jacobian row is `[13 17 19 23]` — every entry
  unique, recognizable on sight in `aug_com`.
- Validates: 4-variable ordering matches the order JuMP variables were
  added; constraint multiplier label `λ[c_primes]` lands in the right row.

### Test structure

Each example becomes a `test_kkt_labels_example_<n>()` function inside
`test/MOI_interface_test.jl` (the same module pattern as other tests).
After `optimize!` with `max_iter=1`:

```julia
solver = unsafe_backend(model).solver
labels = MadNLPMOI.kkt_row_labels(solver)
@test length(labels) == size(solver.kkt.aug_com, 1)
@test labels[1] == "w"   # or whatever the user-set name is
# ... selected coefficient checks via solver.kkt.aug_com
```

Coefficient assertions use `≈` with a generous tolerance (these are
finite-difference / autodiff outputs at the starting point, not exact).

## Out of scope (deferred)

- Using names in MadNLP printer/logger output (option C from brainstorming).
- Per-row names for vector nonlinear oracle constraints (currently same name
  for the whole vector).
- Indexed inverse lookup (currently linear scan; fine for typical sizes).
- Labels for condensed KKT systems (`SparseCondensedKKTSystem`,
  `DenseCondensedKKTSystem`, `ScaledSparseKKTSystem`). Initial implementation
  covers reduced + unreduced; condensed forms can be added later.
