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

## Out of scope (deferred)

- Using names in MadNLP printer/logger output (option C from brainstorming).
- Per-row names for vector nonlinear oracle constraints (currently same name
  for the whole vector).
- Indexed inverse lookup (currently linear scan; fine for typical sizes).
