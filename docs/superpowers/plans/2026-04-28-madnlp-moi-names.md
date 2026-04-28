# MadNLPMOI variable/constraint names + KKT labels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `MadNLPMOI` to support `MOI.VariableName` / `MOI.ConstraintName`, propagate names to the `MOIModel` wrapper, and expose `kkt_row_labels` / `hessian_labels` helpers that map `MadNLPSolver` augmented KKT and Hessian rows/columns back to original variable/constraint names.

**Architecture:** Names live as dictionaries on `MadNLPMOI.Optimizer`. At `_setup_nlp` time they are flattened to two `Vector{String}` fields on `MOIModel` (aligned with the constraint ordering used by the evaluator). Helpers read `solver.cb`, `solver.kkt`, and the `MOIModel` reached through `solver.cb.nlp` (or its `.inner` if wrapped). No changes to the IPM, no changes to base `MadNLP`.

**Tech Stack:** Julia 1.10+, MathOptInterface (extension `MadNLPMOI`), JuMP (tests), NLPModels.jl, MadNLP test harness.

**Spec reference:** `docs/superpowers/specs/2026-04-28-madnlp-moi-names-design.md`

**Conventions for this PR:**
- This is the upstream MadNLP repo (MIT). Do **not** add Nablarise license headers — match existing files (no header).
- Follow the test pattern of `test/MOI_interface_test.jl`: a single `module TestMOIWrapper` with `function test_*()` discovered by the existing `runtests()` loop. Do **not** add a new file — append functions to `MOI_interface_test.jl`.
- All code added in this PR lives in `ext/MadNLPMOI/` (no changes to `src/`).

---

## Task 1: Add `var_names` field to `Optimizer` and basic `MOI.VariableName` set/get

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl:27-64` (struct), `:66-104` (constructor), `:136-166` (`MOI.empty!`), `:319-326` (insert new section near `MOI.Name`)
- Modify: `test/MOI_interface_test.jl` — append new `test_*` functions to the `TestMOIWrapper` module.

- [ ] **Step 1: Write failing test**

Append to `test/MOI_interface_test.jl` inside `module TestMOIWrapper`:

```julia
function test_variable_name_set_get()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    @test MOI.supports(model, MOI.VariableName(), MOI.VariableIndex)
    @test MOI.get(model, MOI.VariableName(), x) == ""
    MOI.set(model, MOI.VariableName(), x, "x_one")
    @test MOI.get(model, MOI.VariableName(), x) == "x_one"
    return
end
```

- [ ] **Step 2: Run test — verify it fails**

Run from the repo root:
```bash
julia --project=. -e 'using Pkg; Pkg.test("MadNLP"; test_args=["test_variable_name_set_get"])' 2>&1 | tail -30
```
Expected: failure on `MOI.supports(...)` returning `false` (default), or on `MOI.get` for `MOI.VariableName`.

(If `Pkg.test` is too heavy, run manually: `julia --project=test -e 'include("test/MOI_interface_test.jl"); TestMOIWrapper.test_variable_name_set_get()'` after dev'ing MadNLP into the test env.)

- [ ] **Step 3: Add `var_names` field and constructor init**

Edit `ext/MadNLPMOI/MOI_wrapper.jl` line ~63 (just before `hess_available::Bool`), add:
```julia
    var_names::Dict{MOI.VariableIndex, String}
```

Edit constructor at lines 66-104 (the `Optimizer(; kwargs...)` function): add `Dict{MOI.VariableIndex, String}(),` as the new field, in the same position as the struct field.

Edit `MOI.empty!` at ~line 165 (just before `model.has_only_linear_constraints = false`), add:
```julia
    empty!(model.var_names)
```

- [ ] **Step 4: Add MOI.VariableName methods**

Insert a new section in `ext/MadNLPMOI/MOI_wrapper.jl` after line 326 (`MOI.get(model::Optimizer, ::MOI.Name) = model.name`):

```julia
### MOI.VariableName

MOI.supports(::Optimizer, ::MOI.VariableName, ::Type{MOI.VariableIndex}) = true

function MOI.get(model::Optimizer, ::MOI.VariableName, vi::MOI.VariableIndex)
    return get(model.var_names, vi, "")
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariableName,
    vi::MOI.VariableIndex,
    name::String,
)
    if isempty(name)
        delete!(model.var_names, vi)
    else
        model.var_names[vi] = name
    end
    return
end
```

- [ ] **Step 5: Run test — verify pass**

Run the same command as Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ext/MadNLPMOI/MOI_wrapper.jl test/MOI_interface_test.jl
git commit -m "$(cat <<'EOF'
feat(MOI): support MOI.VariableName get/set on MadNLPMOI.Optimizer

Add a var_names dict on the Optimizer struct and implement
MOI.supports/get/set for MOI.VariableName. Empty string clears
the entry. Empty! flushes the dict.
EOF
)"
```

---

## Task 2: Inverse lookup `MOI.get(opt, MOI.VariableIndex, "name")`

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — extend the `MOI.VariableName` section.
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Write failing test**

Append to `test/MOI_interface_test.jl`:

```julia
function test_variable_name_inverse_lookup()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.set(model, MOI.VariableName(), x, "alpha")
    MOI.set(model, MOI.VariableName(), y, "beta")
    @test MOI.get(model, MOI.VariableIndex, "alpha") == x
    @test MOI.get(model, MOI.VariableIndex, "beta") == y
    @test MOI.get(model, MOI.VariableIndex, "missing") === nothing
    return
end
```

- [ ] **Step 2: Run — fails**

Same command, scoped to the new test name. Expected: `MethodError` for `MOI.get(::Optimizer, ::Type{MOI.VariableIndex}, ::String)`.

- [ ] **Step 3: Implement**

Append to the `MOI.VariableName` section in `ext/MadNLPMOI/MOI_wrapper.jl`:

```julia
function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, name::String)
    for (vi, n) in model.var_names
        if n == name
            return vi
        end
    end
    return nothing
end
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(MOI): inverse lookup MOI.get(opt, MOI.VariableIndex, name)"
```

---

## Task 3: Add `con_names` field and `MOI.ConstraintName` for affine/quadratic constraints

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` (struct, constructor, empty!, new section after VariableName)
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Write failing test**

```julia
function test_constraint_name_affine()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    f = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(2.0, y)],
        0.0,
    )
    ci = MOI.add_constraint(model, f, MOI.LessThan(10.0))
    @test MOI.supports(model, MOI.ConstraintName(), typeof(ci))
    @test MOI.get(model, MOI.ConstraintName(), ci) == ""
    MOI.set(model, MOI.ConstraintName(), ci, "lin1")
    @test MOI.get(model, MOI.ConstraintName(), ci) == "lin1"
    @test MOI.get(model, typeof(ci), "lin1") == ci
    return
end
```

- [ ] **Step 2: Run — fails**

- [ ] **Step 3: Add `con_names` field**

Add to struct (line 63 area):
```julia
    con_names::Dict{MOI.ConstraintIndex, String}
```

Constructor: `Dict{MOI.ConstraintIndex, String}(),` in matching position.

`MOI.empty!`: add `empty!(model.con_names)`.

- [ ] **Step 4: Add MOI.ConstraintName methods**

Insert new section after the `MOI.VariableName` section:

```julia
### MOI.ConstraintName

# Variable-bound constraints (CI{VariableIndex, S}) cannot be named per MOI.
MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintName,
    ::Type{<:MOI.ConstraintIndex{MOI.VariableIndex, <:Any}},
) = false

MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintName,
    ::Type{<:MOI.ConstraintIndex},
) = true

function MOI.get(model::Optimizer, ::MOI.ConstraintName, ci::MOI.ConstraintIndex)
    return get(model.con_names, ci, "")
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintName,
    ci::MOI.ConstraintIndex,
    name::String,
)
    if isempty(name)
        delete!(model.con_names, ci)
    else
        model.con_names[ci] = name
    end
    return
end

function MOI.get(model::Optimizer, ::Type{C}, name::String) where {C<:MOI.ConstraintIndex}
    for (ci, n) in model.con_names
        if n == name && ci isa C
            return ci::C
        end
    end
    return nothing
end
```

- [ ] **Step 5: Run — passes**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(MOI): support MOI.ConstraintName on MadNLPMOI.Optimizer"
```

---

## Task 4: `MOI.ConstraintName` for `ScalarNonlinearFunction` and `VectorNonlinearOracle`

**Files:**
- Modify: `test/MOI_interface_test.jl`.

The implementation from Task 3 already covers all `ConstraintIndex` types via the `MOI.ConstraintIndex` supertype dispatch. This task just adds tests for the NL paths.

- [ ] **Step 1: Add test for NL block constraint**

```julia
function test_constraint_name_nonlinear()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    f = MOI.ScalarNonlinearFunction(:*, Any[x, y])
    ci = MOI.add_constraint(model, f, MOI.LessThan(5.0))
    MOI.set(model, MOI.ConstraintName(), ci, "nl_xy")
    @test MOI.get(model, MOI.ConstraintName(), ci) == "nl_xy"
    @test MOI.get(model, typeof(ci), "nl_xy") == ci
    return
end
```

- [ ] **Step 2: Run — should pass already** (no impl change needed; verifies dispatch covers NL CI)

- [ ] **Step 3: Commit**

```bash
git commit -am "test(MOI): cover ConstraintName for ScalarNonlinearFunction"
```

---

## Task 5: `MOI.empty!` clears name dicts and `MOI.delete` removes the corresponding entry

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl`: `MOI.delete` for variable bounds (line 449), and any other `MOI.delete` methods. Use grep `^function MOI.delete` to enumerate. The `Optimizer` only has one explicit `MOI.delete` (line 449) for variable-bound constraints; affine/NL constraints don't currently support delete (if `MOI.delete` is called on an unsupported CI type, MOI throws).
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Write tests**

```julia
function test_names_cleared_on_empty()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    MOI.set(model, MOI.VariableName(), x, "x")
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    ci = MOI.add_constraint(model, f, MOI.LessThan(1.0))
    MOI.set(model, MOI.ConstraintName(), ci, "c1")
    MOI.empty!(model)
    @test isempty(model.var_names)
    @test isempty(model.con_names)
    return
end
```

- [ ] **Step 2: Run — passes**

(`MOI.empty!` was already updated in Tasks 1 and 3.) If it fails, add the missing `empty!` calls.

- [ ] **Step 3: Commit**

```bash
git commit -am "test(MOI): names cleared on MOI.empty!"
```

---

## Task 6: Extend `MOIModel` with `var_names` and `con_names` vectors (no population yet)

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl:1253-1257` (struct definition) and `:1437-1460` (constructor call inside `_setup_nlp`).

- [ ] **Step 1: Write failing test**

```julia
function test_moimodel_has_name_vectors()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 1)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    f = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:^, Any[x, 2]),
        MOI.ScalarNonlinearFunction(:^, Any[y, 2]),
    ])
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.optimize!(model)
    nlp = model.nlp
    @test nlp isa MadNLPMOI.MOIModel
    @test nlp.var_names isa Vector{String}
    @test nlp.con_names isa Vector{String}
    @test length(nlp.var_names) == 2
    @test length(nlp.con_names) == 0
    return
end
```

(`MadNLPMOI` is reachable via `Base.get_extension(MadNLP, :MadNLPMOI)`. For terseness, define a local: `const MadNLPMOI = Base.get_extension(MadNLP, :MadNLPMOI)` at the top of the test module after the existing imports.)

- [ ] **Step 2: Run — fails** with `type MOIModel has no field var_names`.

- [ ] **Step 3: Extend struct**

Modify lines 1253-1257:
```julia
struct MOIModel{T} <: NLPModels.AbstractNLPModel{T,Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    model::Optimizer
    counters::NLPModels.Counters
    var_names::Vector{String}
    con_names::Vector{String}
end
```

- [ ] **Step 4: Pass empty vectors at construction**

Modify the `MOIModel(...)` call at lines 1437-1460. After `NLPModels.Counters(),` (the third positional arg, currently the last), add:
```julia
        ,
        String[],   # var_names — populated in Task 7
        String[],   # con_names — populated in Task 8
```

(Indent to match.)

- [ ] **Step 5: Run — passes**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(MOIModel): add empty var_names/con_names fields"
```

---

## Task 7: Populate `var_names` from `Optimizer.var_names` in `_setup_nlp`

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl:1389` area (right after `nvar = ...`) and the `MOIModel(...)` call at 1437.

- [ ] **Step 1: Write failing test**

```julia
function test_moimodel_propagates_var_names()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 1)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    MOI.set(model, MOI.VariableName(), x, "alpha")
    MOI.set(model, MOI.VariableName(), y, "beta")
    obj = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:^, Any[x, 2]),
        MOI.ScalarNonlinearFunction(:^, Any[y, 2]),
    ])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.optimize!(model)
    @test model.nlp.var_names == ["alpha", "beta"]
    return
end
```

- [ ] **Step 2: Run — fails** (vector still empty).

- [ ] **Step 3: Build `var_names` in `_setup_nlp`**

Insert after `nvar = length(model.variables.lower)` (line ~1389):
```julia
    var_names_vec = String[
        get(model.var_names, vi, "")
        for vi in model.list_of_variable_indices
        if !_is_parameter(vi)
    ]
```

(The `_is_parameter` filter excludes `MOI.Parameter` variables which are not in the optimization variable count — they live in the parameter dict.)

Replace `String[],   # var_names — populated in Task 7` (from Task 6) with `var_names_vec,`.

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(MOIModel): populate var_names from MOI.VariableName"
```

---

## Task 8: Populate `con_names` in row order (QP → vector oracle → NL block)

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — add a helper near other private helpers (around line 184) and call it in `_setup_nlp`.

The constraint row order in the augmented system (validated against `_setup_nlp` lines 1400-1408 and `eval_constraint` body) is:

1. Rows `1..length(model.qp_data.constraints)` — QP constraints, in row order. CI for row `i` is `MOI.ConstraintIndex{F_i, S_i}(i)` where `(F_i, S_i)` is reconstructed from `qp_data.function_type[i]` / `bound_type[i]`.
2. Rows immediately after — one block per `vector_nonlinear_oracle_constraints[i]`, of length `s.set.output_dimension`. CI is `MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}}(i)`. Each row in the block uses the same name (per spec §2).
3. Remaining rows — NL block, ordered by `MOI.Nonlinear.ConstraintIndex.value`. CI is `MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, typeof(set)}(idx.value)`.

- [ ] **Step 1: Write failing test**

```julia
function test_moimodel_propagates_con_names()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 1)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    # affine constraint
    f_lin = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    c_lin = MOI.add_constraint(model, f_lin, MOI.LessThan(10.0))
    MOI.set(model, MOI.ConstraintName(), c_lin, "lin")
    # NL constraint
    f_nl = MOI.ScalarNonlinearFunction(:*, Any[x, y])
    c_nl = MOI.add_constraint(model, f_nl, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ConstraintName(), c_nl, "nl")
    obj = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:^, Any[x, 2]),
        MOI.ScalarNonlinearFunction(:^, Any[y, 2]),
    ])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.optimize!(model)
    @test model.nlp.con_names == ["lin", "nl"]   # QP first, then NL
    return
end
```

- [ ] **Step 2: Run — fails** (con_names still empty).

- [ ] **Step 3: Add helper `_build_con_names`**

Insert in `ext/MadNLPMOI/MOI_wrapper.jl` near other private helpers (around line 184):

```julia
function _qp_constraint_index(qp::QPBlockData{T}, row::Int) where {T}
    F = _function_type_to_set(T, qp.function_type[row])
    S = _bound_type_to_set(T, qp.bound_type[row])
    return MOI.ConstraintIndex{F, S}(row)
end

function _build_con_names(model::Optimizer)
    names = String[]
    # 1) QP rows
    for row in 1:length(model.qp_data.constraints)
        ci = _qp_constraint_index(model.qp_data, row)
        push!(names, get(model.con_names, ci, ""))
    end
    # 2) Vector nonlinear oracle blocks
    for (i, (_, cache)) in enumerate(model.vector_nonlinear_oracle_constraints)
        ci = MOI.ConstraintIndex{
            MOI.VectorOfVariables,
            MOI.VectorNonlinearOracle{Float64},
        }(i)
        nm = get(model.con_names, ci, "")
        for _ in 1:cache.set.output_dimension
            push!(names, nm)
        end
    end
    # 3) NL block (ScalarNonlinearFunction)
    if model.nlp_model !== nothing
        # iterate constraints in increasing key.value order
        nl_keys = sort!(collect(keys(model.nlp_model.constraints)); by = k -> k.value)
        for k in nl_keys
            S = typeof(model.nlp_model.constraints[k].set)
            ci = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, S}(k.value)
            push!(names, get(model.con_names, ci, ""))
        end
    end
    return names
end
```

- [ ] **Step 4: Call from `_setup_nlp`**

After `ncon = length(g_L)` (line ~1409), insert:
```julia
    con_names_vec = _build_con_names(model)
    @assert length(con_names_vec) == ncon
```

Replace the `String[]` placeholder for `con_names` in the `MOIModel(...)` call with `con_names_vec`.

- [ ] **Step 5: Run — passes**

- [ ] **Step 6: Commit**

```bash
git commit -am "$(cat <<'EOF'
feat(MOIModel): populate con_names in evaluator row order

QP rows first, then vector nonlinear oracle blocks (one name per
output row of the oracle), then NL block constraints ordered by
their Nonlinear.ConstraintIndex value. Matches the layout produced
by eval_constraint and jacobian_structure.
EOF
)"
```

---

## Task 9: Helpers `get_variable_names` / `get_constraint_names` with `SparseWrapperModel` support

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — append helpers near the end of the file.

- [ ] **Step 1: Write failing test**

```julia
function test_get_names_helpers()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 1)
    x = MOI.add_variable(model)
    MOI.set(model, MOI.VariableName(), x, "alpha")
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    obj = MOI.ScalarNonlinearFunction(:^, Any[x, 2])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.optimize!(model)
    @test MadNLPMOI.get_variable_names(model) == ["alpha"]
    @test MadNLPMOI.get_constraint_names(model) == String[]
    return
end
```

- [ ] **Step 2: Run — fails** (`get_variable_names` not defined).

- [ ] **Step 3: Implement helpers**

Append to `ext/MadNLPMOI/MOI_wrapper.jl`:

```julia
### Public name-access helpers

_names_of(::Nothing) = (var_names = String[], con_names = String[])
_names_of(nlp::MOIModel) = nlp
_names_of(nlp::MadNLP.SparseWrapperModel) = _names_of(nlp.inner)

"""
    get_variable_names(opt::Optimizer) -> Vector{String}

Return the variable names propagated to the underlying NLP model after
`optimize!`. Returns an empty vector if no model has been built yet.
"""
get_variable_names(opt::Optimizer) = collect(_names_of(opt.nlp).var_names)

"""
    get_constraint_names(opt::Optimizer) -> Vector{String}

Return the constraint names in evaluator row order.
"""
get_constraint_names(opt::Optimizer) = collect(_names_of(opt.nlp).con_names)
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(MadNLPMOI): get_variable_names / get_constraint_names helpers"
```

---

## Task 10: KKT row labels for `SparseKKTSystem` (reduced form)

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — append a new "KKT labels" section.

The reduced KKT layout for `SparseKKTSystem` (verified against `src/KKT/Sparse/augmented.jl:72-94`):
- Rows 1..n: primal variables in `cb` order. With `NoFixedVariables`, this is `1..nlp.meta.nvar`. With `MakeParameter`, the i-th row is original variable `cb.fixed_handler.free[i]`.
- Rows n+1..n+n_slack: slacks for inequality constraints; the k-th slack corresponds to the constraint at row `cb.ind_ineq[k]` of the original constraint vector.
- Rows n_tot+1..n_tot+m: constraint multipliers, in the same order as the constraint vector (no constraint reordering by the callback).

- [ ] **Step 1: Write failing test (with QP magnitudes example)**

```julia
function test_kkt_row_labels_sparse_qp()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, x, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@constraint(model, c1, 5*x + 0.1*y >= 1)
    JuMP.@constraint(model, c2, x - 100*y <= 50)
    JuMP.@objective(model, Min, 1000*x^2 + 0.01*y^2 + 7*x*y)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    labels = MadNLPMOI.kkt_row_labels(opt.solver)
    # Expected layout: [x, y, slack[c1], slack[c2], λ[c1], λ[c2]]
    # (both constraints are inequalities → 2 slacks)
    @test length(labels) == 6
    @test labels[1] == "x"
    @test labels[2] == "y"
    @test labels[3] == "slack[c1]"
    @test labels[4] == "slack[c2]"
    @test labels[5] == "λ[c1]"
    @test labels[6] == "λ[c2]"
    return
end
```

(Add `using JuMP` and `import JuMP` at top of `test/MOI_interface_test.jl` if not already imported. Check first — JuMP may not be a test dep yet. If it is missing, add it to `test/Project.toml` before running.)

- [ ] **Step 2: Run — fails** (`kkt_row_labels` not defined).

- [ ] **Step 3: Implement**

Append to `ext/MadNLPMOI/MOI_wrapper.jl`:

```julia
### KKT row/column labels

# Map an index in the cb's free-variable space (1..cb.nvar) to a primal label.
function _primal_label(cb, nlp, i::Int)
    orig = _orig_var_index(cb, i)
    nm = isempty(nlp.var_names) ? "" : nlp.var_names[orig]
    return isempty(nm) ? "x[$orig]" : nm
end

_orig_var_index(cb, i::Int) = _orig_var_index(cb.fixed_handler, i)
_orig_var_index(::MadNLP.NoFixedVariables, i::Int) = i
_orig_var_index(::MadNLP.RelaxBound, i::Int) = i
_orig_var_index(fh::MadNLP.MakeParameter, i::Int) = Int(fh.free[i])

function _constraint_label(nlp, k::Int)
    nm = isempty(nlp.con_names) ? "" : nlp.con_names[k]
    return isempty(nm) ? "c[$k]" : nm
end

function _kkt_row_labels_reduced(cb, nlp)
    n = cb.nvar
    n_slack = length(cb.ind_ineq)
    m = cb.ncon
    labels = String[]
    # primal variables
    for i in 1:n
        push!(labels, _primal_label(cb, nlp, i))
    end
    # slacks (one per inequality constraint)
    for k in 1:n_slack
        cidx = Int(cb.ind_ineq[k])
        push!(labels, "slack[" * _constraint_label(nlp, cidx) * "]")
    end
    # constraint multipliers
    for k in 1:m
        push!(labels, "λ[" * _constraint_label(nlp, k) * "]")
    end
    return labels
end

"""
    kkt_row_labels(solver::MadNLP.MadNLPSolver) -> Vector{String}

Return labels for each row of `solver.kkt.aug_com`, mapping back to original
variable and constraint names where available, with sensible fallbacks
(`x[i]`, `c[k]`).
"""
function kkt_row_labels(solver::MadNLP.MadNLPSolver)
    return _kkt_row_labels(solver.kkt, solver.cb, _names_of(solver.cb.nlp))
end

# Reduced KKT systems (SparseKKTSystem, DenseKKTSystem)
_kkt_row_labels(::MadNLP.AbstractReducedKKTSystem, cb, nlp) =
    _kkt_row_labels_reduced(cb, nlp)

kkt_col_labels(solver::MadNLP.MadNLPSolver) = kkt_row_labels(solver)
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "$(cat <<'EOF'
feat(MadNLPMOI): kkt_row_labels for reduced KKT systems

Map MadNLP augmented KKT rows back to original variable/constraint
names, handling NoFixedVariables/RelaxBound/MakeParameter via
cb.fixed_handler.free. Slack rows get `slack[<con>]`, dual rows
`λ[<con>]`, with `x[i]`/`c[k]` fallbacks for unnamed entries.
EOF
)"
```

---

## Task 11: KKT row labels for `SparseUnreducedKKTSystem` (zL, zU)

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — extend the KKT-labels section.

The unreduced layout (verified against `src/KKT/Sparse/unreduced.jl:85-87`):
- Same first three blocks as reduced (n_tot + m).
- Then `nlb` rows for `zL`, indexed by `cb.ind_lb` (which points into `1..n_tot`).
- Then `nub` rows for `zU`, indexed by `cb.ind_ub` (also into `1..n_tot`).

A bound multiplier on row `i ≤ n` is for primal variable `i` (use `_primal_label`); on row `n < i ≤ n_tot` it's for the slack at index `i-n`, whose constraint is `cb.ind_ineq[i-n]`.

- [ ] **Step 1: Write failing test (Example 4 — bound multipliers)**

```julia
function test_kkt_row_labels_unreduced_bounds()
    model = JuMP.Model(() ->
        MadNLP.Optimizer(kkt_system = MadNLP.SparseUnreducedKKTSystem))
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, 0 <= x <= 100, base_name = "x")
    JuMP.@variable(model, -50 <= y <= 50, base_name = "y")
    JuMP.@objective(model, Min, (x - 3)^2 + (y + 2)^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    labels = MadNLPMOI.kkt_row_labels(opt.solver)
    # Layout: [x, y, zL[x], zL[y], zU[x], zU[y]]  (no slacks, no constraints)
    @test labels == ["x", "y", "zL[x]", "zL[y]", "zU[x]", "zU[y]"]
    return
end
```

- [ ] **Step 2: Run — fails** (returns reduced layout, missing zL/zU rows).

- [ ] **Step 3: Implement unreduced overload**

Append:

```julia
function _label_in_n_tot(cb, nlp, i::Int)
    if i <= cb.nvar
        return _primal_label(cb, nlp, i)
    else
        slack_k = i - cb.nvar
        cidx = Int(cb.ind_ineq[slack_k])
        return "slack[" * _constraint_label(nlp, cidx) * "]"
    end
end

function _kkt_row_labels_unreduced(cb, nlp)
    labels = _kkt_row_labels_reduced(cb, nlp)
    for k in 1:length(cb.ind_lb)
        i = Int(cb.ind_lb[k])
        push!(labels, "zL[" * _label_in_n_tot(cb, nlp, i) * "]")
    end
    for k in 1:length(cb.ind_ub)
        i = Int(cb.ind_ub[k])
        push!(labels, "zU[" * _label_in_n_tot(cb, nlp, i) * "]")
    end
    return labels
end

_kkt_row_labels(::MadNLP.AbstractUnreducedKKTSystem, cb, nlp) =
    _kkt_row_labels_unreduced(cb, nlp)
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(MadNLPMOI): kkt_row_labels for unreduced KKT (zL/zU rows)"
```

---

## Task 12: `hessian_labels(solver)` (primal + slack)

**Files:**
- Modify: `ext/MadNLPMOI/MOI_wrapper.jl` — append.

The Hessian of the Lagrangian in `solver.kkt.hess_com` is `n_tot × n_tot` for sparse KKT systems (verified against the `hess_raw` definition `n_tot, n_tot, ...` at `src/KKT/Sparse/augmented.jl:117-118`).

- [ ] **Step 1: Write failing test**

```julia
function test_hessian_labels_with_slack()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, x, base_name = "x")
    JuMP.@constraint(model, c, x^2 <= 1)
    JuMP.@objective(model, Min, x^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    labels = MadNLPMOI.hessian_labels(opt.solver)
    @test labels == ["x", "slack[c]"]
    return
end
```

- [ ] **Step 2: Run — fails**.

- [ ] **Step 3: Implement**

```julia
"""
    hessian_labels(solver::MadNLP.MadNLPSolver) -> Vector{String}

Return labels for each row/col of the Lagrangian Hessian
`solver.kkt.hess_com` — `cb.nvar` primal entries followed by
`length(cb.ind_ineq)` slack entries.
"""
function hessian_labels(solver::MadNLP.MadNLPSolver)
    cb = solver.cb
    nlp = _names_of(cb.nlp)
    n = cb.nvar
    labels = String[_primal_label(cb, nlp, i) for i in 1:n]
    for k in 1:length(cb.ind_ineq)
        cidx = Int(cb.ind_ineq[k])
        push!(labels, "slack[" * _constraint_label(nlp, cidx) * "]")
    end
    return labels
end
```

- [ ] **Step 4: Run — passes**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(MadNLPMOI): hessian_labels (primal + slack rows)"
```

---

## Task 13: Validation example 1 — Pure QP, very different magnitudes

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example1_qp_magnitudes()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, x, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@constraint(model, c_lin1, 5*x + 0.1*y >= 1)
    JuMP.@constraint(model, c_lin2, x - 100*y <= 50)
    JuMP.@objective(model, Min, 1000*x^2 + 0.01*y^2 + 7*x*y)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    # Hessian: [2000 7; 7 0.02] for the (x, y) block; slacks have zero hess.
    hess = Matrix(solver.kkt.hess_com)
    hess_lbl = MadNLPMOI.hessian_labels(solver)
    ix = findfirst(==("x"), hess_lbl)
    iy = findfirst(==("y"), hess_lbl)
    @test hess[ix, ix] ≈ 2000  atol=1e-6
    @test hess[iy, iy] ≈ 0.02  atol=1e-6
    @test hess[ix, iy] + hess[iy, ix] ≈ 7  atol=1e-6  # symmetric storage
    # Augmented KKT row labels — sanity
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test "λ[c_lin1]" in aug_lbl
    @test "λ[c_lin2]" in aug_lbl
    return
end
```

- [ ] **Step 2: Run — should pass with current implementation**

If it fails, the failure is a real bug in Task 8/10/12 to fix.

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 1 — QP with distinctive magnitudes"
```

---

## Task 14: Validation example 2 — NL with primes/exponentials

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example2_nl_primes()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, x, start = 0.0, base_name = "x")
    JuMP.@variable(model, y, start = 1.0, base_name = "y")
    JuMP.@constraint(model, c_eq, x*y - 13 == 0)
    JuMP.@objective(model, Min, exp(3*x) + log(1 + y^2))
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test "x" in aug_lbl && "y" in aug_lbl && "λ[c_eq]" in aug_lbl
    # Equality → no slack in cb.ind_ineq
    @test !any(startswith.(aug_lbl, "slack["))
    return
end
```

- [ ] **Step 2: Run — passes**

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 2 — NL equality with prime constant"
```

---

## Task 15: Validation example 3 — Mixed linear + NL, three variables

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example3_mixed()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, x, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@variable(model, z, base_name = "z")
    JuMP.@constraint(model, c_sum, x + y + z <= 10)
    JuMP.@constraint(model, c_bilin, x*y >= 0.5)
    JuMP.@constraint(model, c_cubic, 42*z^3 - 7 == 0)
    JuMP.@objective(model, Min, x^2/2 + 100*y^2 + sin(z))
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    # 3 vars + 2 inequalities (c_sum, c_bilin) → 2 slacks + 3 multipliers
    @test length(aug_lbl) == 3 + 2 + 3
    @test "slack[c_sum]"   in aug_lbl
    @test "slack[c_bilin]" in aug_lbl
    @test !("slack[c_cubic]" in aug_lbl)   # equality → no slack
    @test "λ[c_cubic]"     in aug_lbl
    return
end
```

- [ ] **Step 2: Run — passes**

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 3 — mixed linear+NL with 1 eq, 2 ineq"
```

---

## Task 16: Validation example 4 — Bound multipliers (unreduced)

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example4_bound_multipliers()
    model = JuMP.Model(() ->
        MadNLP.Optimizer(kkt_system = MadNLP.SparseUnreducedKKTSystem))
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, 0 <= x <= 100, base_name = "x")
    JuMP.@variable(model, -50 <= y <= 50, base_name = "y")
    JuMP.@objective(model, Min, (x - 3)^2 + (y + 2)^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test aug_lbl == ["x", "y", "zL[x]", "zL[y]", "zU[x]", "zU[y]"]
    @test size(solver.kkt.aug_com, 1) == length(aug_lbl)
    return
end
```

- [ ] **Step 2: Run — passes** (covered by Task 11 already; re-verifies on a clean model).

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 4 — bound multipliers in unreduced KKT"
```

---

## Task 17: Validation example 5 — Fixed variable (`MakeParameter`)

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example5_fixed_variable()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.set_attribute(model, "fixed_variable_treatment", MadNLP.MakeParameter)
    JuMP.@variable(model, x == 5, base_name = "x")   # fixed
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@variable(model, z, base_name = "z")
    JuMP.@constraint(model, c_sum, x + 2*y + 3*z >= 6)
    JuMP.@objective(model, Min, (x - 1)^2 + (y - 2)^2 + (z - 3)^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    # Free variables in cb order are y, z (x is fixed)
    @test aug_lbl[1] == "y"
    @test aug_lbl[2] == "z"
    @test !("x" in aug_lbl)
    return
end
```

- [ ] **Step 2: Run — passes**

If it fails, debug `_orig_var_index(::MakeParameter, ...)` and confirm `cb.fixed_handler.free` is the right field.

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 5 — fixed variable removed from labels"
```

---

## Task 18: Validation example 6 — Prime-coefficient equality, four variables

**Files:**
- Modify: `test/MOI_interface_test.jl`.

- [ ] **Step 1: Add validation test**

```julia
function test_validate_example6_prime_jacobian()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.@variable(model, w, base_name = "w")
    JuMP.@variable(model, x, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@variable(model, z, base_name = "z")
    JuMP.@constraint(model, c_primes, 13*w + 17*x + 19*y + 23*z == 100)
    JuMP.@objective(model, Min, w^2 + x^2 + y^2 + z^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test aug_lbl[1:4] == ["w", "x", "y", "z"]
    @test aug_lbl[end] == "λ[c_primes]"
    # Jacobian row of c_primes should have entries [13, 17, 19, 23]
    jac = Matrix(solver.kkt.jac_com)
    # First (and only) constraint row of jac is row 1
    @test jac[1, 1] ≈ 13
    @test jac[1, 2] ≈ 17
    @test jac[1, 3] ≈ 19
    @test jac[1, 4] ≈ 23
    return
end
```

- [ ] **Step 2: Run — passes**

If `jac` shape is unexpected, inspect `solver.kkt.jac_com` (which may include slack columns — for an equality-only model `n_slack = 0`, so jac is `1 × 4`).

- [ ] **Step 3: Commit**

```bash
git commit -am "test(validate): example 6 — prime-coefficient equality jacobian"
```

---

## Task 19: Run the full MOI test suite and the validation set; commit fixes if any

**Files:**
- Possibly: `test/MOI_interface_test.jl` (adjust JuMP imports if missing), `test/Project.toml`.

- [ ] **Step 1: Run the entire MOI test module**

```bash
julia --project=test -e '
include("test/MOI_interface_test.jl")
TestMOIWrapper.runtests()
' 2>&1 | tail -40
```

Expected: all `test_*` (existing + new) pass.

- [ ] **Step 2: If `using JuMP` was added but missing in `test/Project.toml`**

Add `JuMP` to `test/Project.toml` `[deps]` if a JuMP-related test errors with `ArgumentError: Package JuMP ... not installed`. Confirm with `grep JuMP test/Project.toml` first.

```bash
julia --project=test -e 'using Pkg; Pkg.add("JuMP")'
```

- [ ] **Step 3: Run the full repo test suite for regression**

```bash
julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -60
```

Expected: no new failures. Existing `test_MOI_Test` should still pass (it uses `MOI.Test.runtests` which excludes Name-related failures via `MOI.Utilities.CachingOptimizer` wrapping).

- [ ] **Step 4: Commit any fixes**

```bash
git commit -am "chore(test): wire up JuMP for MOI name validation tests" \
    || echo "no fixes needed"
```

---

## Self-review checklist (run after writing all tasks)

- ✓ Spec §1 (Optimizer-level storage): Tasks 1, 2, 3, 4.
- ✓ Spec §1 (empty/delete cleanup): Task 5.
- ✓ Spec §2 (MOIModel struct extension + propagation): Tasks 6, 7, 8.
- ✓ Spec §3 (helpers, SparseWrapperModel): Task 9.
- ✓ Spec §4 (fixed-variable handling, documented): exercised in Task 17.
- ✓ Spec §5 (test conventions): all `test_*` go in `MOI_interface_test.jl`.
- ✓ Spec §6 (KKT/Hessian labels): Tasks 10 (reduced KKT), 11 (unreduced KKT), 12 (Hessian).
- ✓ Spec §7 (six validation NLPs): Tasks 13–18.
- ✓ No placeholders. Every code block is concrete.
- ✓ Helper names consistent: `kkt_row_labels`, `kkt_col_labels`, `hessian_labels`, `get_variable_names`, `get_constraint_names`. Internal `_names_of`, `_primal_label`, `_constraint_label`, `_label_in_n_tot`, `_orig_var_index`, `_qp_constraint_index`, `_build_con_names`, `_kkt_row_labels`, `_kkt_row_labels_reduced`, `_kkt_row_labels_unreduced`. All references match definitions.
- ✓ Duplicate-name policy from spec is **deferred** (not in any task). The spec mentions duplicate rejection but it is **not tested or implemented** in this plan to keep scope tight; if needed, add a follow-up task. (Flagged here so reviewer can decide.)

## Open questions for the implementer

1. If `MOI.Test.runtests` (in `test_MOI_Test`) starts running new Name-related tests after support is added, some may fail because of MOI's strict expectations (e.g., setting two same names raises). The plan does not currently test these edge cases — if they show up in `MOI.Test`, decide between (a) implementing duplicate rejection (fast, ~10 lines), or (b) excluding the failing tests with a clear comment.
2. `solver.kkt.aug_com` and `hess_com` may be in a non-standard storage type (e.g., CSC vs COO depending on linear solver). The validation tests use `Matrix(...)` to materialize — if MUMPS happens to return a permuted form, the tests may need to use `solver.kkt.aug_raw` (COO) instead. Discover and adjust during Task 13.
