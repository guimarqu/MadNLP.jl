### KKT diagnostic helper

import LinearAlgebra: cond, svd

"""
    KKTDiagnostic

Snapshot of an `MadNLPSolver` augmented KKT system, with labeled magnitudes
and a smallest-singular-vector breakdown. Returned by `kkt_diagnostic`.

Fields are exposed for programmatic inspection. A `Base.show` method
pretty-prints the snapshot to make it readable from the REPL.
"""
struct KKTDiagnostic
    iter::Int
    aug_size::Tuple{Int,Int}
    cond_aug::Float64
    sigma_max::Float64
    sigma_min::Float64
    aug_max::Float64
    aug_min_nz::Float64
    # Hessian diagonal entries [(label, value)]
    hess_diag::Vector{Tuple{String,Float64}}
    hess_spread::Float64
    # Jacobian per-row spread [(label, min_nonzero, max)]
    jac_row_spread::Vector{Tuple{String,Float64,Float64}}
    # Primal magnitudes [(label, value)] — length = cb.nvar + n_slack
    primal::Vector{Tuple{String,Float64}}
    primal_spread::Float64
    # Constraint multipliers [(label, value)]
    dual_con::Vector{Tuple{String,Float64}}
    dual_con_spread::Float64
    # Bound multipliers (only nonempty rows kept) [(label, value)]
    dual_lb::Vector{Tuple{String,Float64}}
    dual_ub::Vector{Tuple{String,Float64}}
    # Smallest singular vector — top contributors [(label, component)]
    soft_mode::Vector{Tuple{String,Float64}}
end

"""
    kkt_diagnostic(solver::MadNLP.MadNLPSolver; n_top::Int = 5) -> KKTDiagnostic

Build a labeled diagnostic snapshot of `solver.kkt.aug_com` (augmented KKT)
and `solver.kkt.hess_com` / `solver.kkt.jac_com`. Reports conditioning,
diagonal/row spreads, primal and dual magnitudes (with original
variable/constraint names where available), and the top `n_top` contributors
to the smallest singular vector.

Use right after `optimize!` (with a small `max_iter` if you want to inspect
mid-iteration) on a model built with names. Works with any KKT system type
that exposes `aug_com`, `hess_com`, and `jac_com`.

Note: computes a dense `cond` and `svd` of the augmented KKT, so it is
intended for debugging on small/medium problems, not as a production hook.
"""
function kkt_diagnostic(solver::MadNLP.MadNLPSolver; n_top::Int = 5)
    cb  = solver.cb
    nlp = _names_of(cb.nlp)

    aug = Matrix(solver.kkt.aug_com)
    hess = Matrix(solver.kkt.hess_com)
    jac  = Matrix(solver.kkt.jac_com)

    labels = kkt_row_labels(solver)
    hess_lbl = hessian_labels(solver)

    n_aug = size(aug, 1)
    nz = filter(!iszero, aug)
    aug_min_nz = isempty(nz) ? 0.0 : minimum(abs, nz)
    aug_max = maximum(abs, aug)

    U, S, V = svd(aug)
    sigma_max = S[1]
    sigma_min = S[end]
    cond_aug = sigma_max / max(sigma_min, eps(Float64))

    # Hessian diagonal
    hess_diag = [(hess_lbl[i], hess[i, i]) for i in 1:size(hess, 1)]
    hd_nz = filter(x -> abs(x[2]) > 0, hess_diag)
    hess_spread = isempty(hd_nz) ? 0.0 :
        maximum(abs, last.(hd_nz)) / minimum(abs, last.(hd_nz))

    # Jacobian per-row spread
    jac_row_spread = Tuple{String,Float64,Float64}[]
    n_qp = length(cb.nlp.model.qp_data.constraints)
    for k in 1:size(jac, 1)
        row = jac[k, :]
        nz = filter(!iszero, row)
        lbl = _constraint_label(nlp, k)
        if isempty(nz)
            push!(jac_row_spread, (lbl, 0.0, 0.0))
        else
            push!(jac_row_spread,
                  (lbl, minimum(abs, nz), maximum(abs, nz)))
        end
    end

    # Primal magnitudes (variables in cb order, then slacks)
    xs = solver.x.values
    primal = [(hess_lbl[i], xs[i]) for i in 1:length(xs)]
    p_nz = filter(x -> abs(x[2]) > 0, primal)
    primal_spread = isempty(p_nz) ? 0.0 :
        maximum(abs, last.(p_nz)) / minimum(abs, last.(p_nz))

    # Constraint multipliers
    y = solver.y
    dual_con = [(_constraint_label(nlp, k), y[k]) for k in 1:length(y)]
    d_nz = filter(x -> abs(x[2]) > 0, dual_con)
    dual_con_spread = isempty(d_nz) ? 0.0 :
        maximum(abs, last.(d_nz)) / minimum(abs, last.(d_nz))

    # Bound multipliers — labels via _label_in_n_tot since ind_lb/ind_ub
    # point into the n_tot space (free vars + slacks)
    dual_lb = Tuple{String,Float64}[]
    for k in 1:length(cb.ind_lb)
        i = Int(cb.ind_lb[k])
        push!(dual_lb, (_label_in_n_tot(cb, nlp, i), solver.zl_r[k]))
    end
    dual_ub = Tuple{String,Float64}[]
    for k in 1:length(cb.ind_ub)
        i = Int(cb.ind_ub[k])
        push!(dual_ub, (_label_in_n_tot(cb, nlp, i), solver.zu_r[k]))
    end

    # Smallest singular vector
    v_min = V[:, end]
    order = sortperm(abs.(v_min); rev = true)
    soft_mode = [(labels[k], v_min[k]) for k in order[1:min(n_top, n_aug)]]

    return KKTDiagnostic(
        solver.cnt.k,
        size(aug),
        cond_aug,
        sigma_max,
        sigma_min,
        aug_max,
        aug_min_nz,
        hess_diag,
        hess_spread,
        jac_row_spread,
        primal,
        primal_spread,
        dual_con,
        dual_con_spread,
        dual_lb,
        dual_ub,
        soft_mode,
    )
end

function _hsep(io::IO)
    println(io, "─"^72)
end

function Base.show(io::IO, ::MIME"text/plain", d::KKTDiagnostic)
    println(io, "KKTDiagnostic @ iter $(d.iter)")
    _hsep(io)
    println(io, "Augmented KKT  : $(d.aug_size[1]) × $(d.aug_size[2])")
    @printf(io, "  cond           = %12.3e\n", d.cond_aug)
    @printf(io, "  σ_max          = %12.3e\n", d.sigma_max)
    @printf(io, "  σ_min          = %12.3e\n", d.sigma_min)
    @printf(io, "  |aug| max      = %12.3e\n", d.aug_max)
    @printf(io, "  |aug| min ≠ 0  = %12.3e\n", d.aug_min_nz)

    _hsep(io)
    @printf(io, "Hessien (diag)   spread |max|/|min| = %.3e\n", d.hess_spread)
    for (lbl, v) in d.hess_diag
        @printf(io, "  H[%-25s] = %+12.3e\n", lbl, v)
    end

    _hsep(io)
    println(io, "Jacobien — étendue par ligne (|min ≠ 0| .. |max|)")
    for (lbl, lo, hi) in d.jac_row_spread
        spread = lo > 0 ? hi / lo : 0.0
        @printf(io, "  jac[%-15s] : %.3e .. %.3e   (spread × %.3e)\n",
                lbl, lo, hi, spread)
    end

    _hsep(io)
    @printf(io, "Primaux (x ∪ s)  spread |max|/|min| = %.3e\n", d.primal_spread)
    for (lbl, v) in d.primal
        @printf(io, "  %-30s = %+12.3e\n", lbl, v)
    end

    _hsep(io)
    @printf(io, "Duals contraintes  spread |max|/|min| = %.3e\n", d.dual_con_spread)
    for (lbl, v) in d.dual_con
        @printf(io, "  λ[%-15s] = %+12.3e\n", lbl, v)
    end

    if !isempty(d.dual_lb) || !isempty(d.dual_ub)
        _hsep(io)
        println(io, "Multiplicateurs de bornes")
        for (lbl, v) in d.dual_lb
            @printf(io, "  zL[%-25s] = %+12.3e\n", lbl, v)
        end
        for (lbl, v) in d.dual_ub
            @printf(io, "  zU[%-25s] = %+12.3e\n", lbl, v)
        end
    end

    _hsep(io)
    println(io, "Mode mou (vecteur singulier σ_min) — top $(length(d.soft_mode)) :")
    for (lbl, v) in d.soft_mode
        @printf(io, "  %-30s : %+9.3e\n", lbl, v)
    end
    return
end

Base.show(io::IO, d::KKTDiagnostic) = print(io, "KKTDiagnostic(iter=$(d.iter), cond=$(round(d.cond_aug; sigdigits=3)))")
