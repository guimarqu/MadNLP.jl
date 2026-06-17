module TestMOIWrapper

using MadNLP
using Test

import JuMP
using MathOptInterface
const MOI = MathOptInterface
const MadNLPMOI = Base.get_extension(MadNLP, :MadNLPMOI)

function runtests()
    for name in names(@__MODULE__; all=true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MadNLP.Optimizer(),
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol=1e-4,
            rtol=1e-4,
            infeasible_status=MOI.LOCALLY_INFEASIBLE,
            optimal_status=MOI.LOCALLY_SOLVED,
            exclude=Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ]
        );
        exclude = [
            # MadNLP reaches maximum number of iterations instead
            # of returning infeasibility certificate.
            r"test_linear_DUAL_INFEASIBLE.*",
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # Symbolic exception in Mumps
            "test_solve_VariableIndex_ConstraintDual_",
            # Tests excluded on purpose
            # - Excluded because Hessian information is needed
            "test_nonlinear_hs071_hessian_vector_product",
            # - Excluded because Hessian information is needed
            "test_nonlinear_invalid",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            # Throw an error: "Unable to query the dual of a variable
            # bound that was reformulated using `ZerosBridge`."
            "test_linear_VectorAffineFunction_empty_row",
            "test_conic_linear_VectorOfVariables_2",
            # TODO: investigate why it is breaking.
            "test_nonlinear_expression_hs109",
        ]
    )

    return
end

function test_extra()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("linear_solver"), UmfpackSolver)

    @test MOI.supports(model, MOI.Name())
    @test MOI.get(model, MOI.Name()) == ""
    MOI.set(model, MOI.Name(), "Model")
    @test MOI.get(model, MOI.Name()) == "Model"

    @test MOI.get(model, MOI.BarrierIterations()) == 0

    return
end

# See issue #239 (https://github.com/MadNLP/MadNLP.jl/issues/239)
function test_invalid_number_in_hessian_lagrangian()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    nlp = MOI.Nonlinear.Model()
    MOI.Nonlinear.set_objective(nlp, :(($x - 5)^2 + ($y - 8)^2))
    MOI.Nonlinear.add_constraint(nlp, :($x * $y), MOI.EqualTo(5.0))
    ev = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), [x, y])
    MOI.set(model, MOI.NLPBlock(), MOI.NLPBlockData(ev))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    return
end

# See issue #318
function test_user_defined_function()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    # Define custom function.
    f(a, b) = a^2 + b^2
    x = MOI.add_variables(model, 2)
    MOI.set(model, MOI.UserDefinedFunction(:f, 2), (f,))
    obj_f = MOI.ScalarNonlinearFunction(:f, Any[x[1], x[2]])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj_f)}(), obj_f)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
end

# See PR #379 (example 1)
function test_param_in_quadratic_term1()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 2.0))
    y, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 2.0))
    a, _ = MOI.add_constrained_variable(model, MOI.Parameter(3.0))
    b, _ = MOI.add_constrained_variable(model, MOI.Parameter(3.0))

    # constraint : a*x^2 + b*y^2 <= 1
    sq_x = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, x)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0
    )
    sq_y = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, y, y)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0
    )
    x_term = MOI.ScalarNonlinearFunction(:*, [a, sq_x])
    y_term = MOI.ScalarNonlinearFunction(:*, [b, sq_y])
    x_y_sum = MOI.ScalarNonlinearFunction(:+, [x_term, y_term])
    lhs = MOI.ScalarNonlinearFunction(:-, [x_y_sum, 1])
    c = MOI.add_constraint(model, lhs, MOI.LessThan{Float64}(0.0))

    # objective function : x + y
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    obj_terms = [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)]
    obj_f = MOI.ScalarAffineFunction(obj_terms, 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj_f)}(), obj_f)

    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    x_val = MOI.get(model, MOI.VariablePrimal(), x)
    y_val = MOI.get(model, MOI.VariablePrimal(), y)
    @test abs(3 * x_val^2 + 3 * y_val^2 - 1) <= 1e-6 # constraint has no slack
end

# See PR #379 (example 2)
function test_param_in_quadratic_term2()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    x, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 2.0))
    y, _ = MOI.add_constrained_variable(model, MOI.Interval(0.0, 2.0))
    a, _ = MOI.add_constrained_variable(model, MOI.Parameter(3.0))
    b, _ = MOI.add_constrained_variable(model, MOI.Parameter(3.0))

    # constraint:  a*x + b^2*y + a*b^2 <= 42 => 3*x + 9*y <= 42 - 3*9
    sq_b = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, b, b)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0
    )

    term1 = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(1.0, a, x)],
        MOI.ScalarAffineTerm{Float64}[],
        0.0
    )
    term2 = MOI.ScalarNonlinearFunction(:*, [sq_b, y])
    term3 = MOI.ScalarNonlinearFunction(:*, [a, sq_b])

    lhs = MOI.ScalarNonlinearFunction(:+, [term1, term2, term3])
    c = MOI.add_constraint(model, lhs, MOI.LessThan{Float64}(42.0))

    # objective function : x + y
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    obj_terms = [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)]
    obj_f = MOI.ScalarAffineFunction(obj_terms, 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj_f)}(), obj_f)

    MOI.optimize!(model)
    @assert MOI.get(model, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
    x_val = MOI.get(model, MOI.VariablePrimal(), x)
    y_val = MOI.get(model, MOI.VariablePrimal(), y)
    @assert abs(3 * x_val + 9 * y_val - 15) <= 1e-6 # constraint has no slack
end

function test_parameter_is_valid()
    model = MadNLP.Optimizer()
    p, ci = MOI.add_constrained_variable(model, MOI.Parameter(2.0))
    @test MOI.is_valid(model, p)
    @test MOI.is_valid(model, ci)
    @test !MOI.is_valid(model, typeof(p)(p.value + 1))
    @test !MOI.is_valid(model, typeof(ci)(ci.value + 1))
    return
end

function test_Parameter_basic()
    F, S = MOI.VariableIndex, MOI.Parameter{Float64}
    model = MadNLP.Optimizer()
    @test MOI.supports_add_constrained_variable(model, S)
    @test !MOI.supports_constraint(model, F, S)
    @test isempty(MOI.get(model, MOI.ListOfConstraintTypesPresent()))
    p1, c1 = MOI.add_constrained_variable(model, MOI.Parameter(1.0))
    @test MOI.is_valid(model, c1)
    @test (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 1
    @test MOI.get(model, MOI.ListOfConstraintIndices{F,S}()) == [c1]
    p2, c2 = MOI.add_constrained_variable(model, MOI.Parameter(2.0))
    @test MOI.get(model, MOI.NumberOfConstraints{F,S}()) == 2
    @test MOI.get(model, MOI.ListOfConstraintIndices{F,S}()) == [c1, c2]
    return
end

function test_variable_name_set_get()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    @test MOI.supports(model, MOI.VariableName(), MOI.VariableIndex)
    @test MOI.get(model, MOI.VariableName(), x) == ""
    MOI.set(model, MOI.VariableName(), x, "x_one")
    @test MOI.get(model, MOI.VariableName(), x) == "x_one"
    return
end

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

function test_variable_name_clear_on_empty_string()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    MOI.set(model, MOI.VariableName(), x, "x")
    @test MOI.get(model, MOI.VariableName(), x) == "x"
    MOI.set(model, MOI.VariableName(), x, "")
    @test MOI.get(model, MOI.VariableName(), x) == ""
    @test isempty(model.var_names)
    return
end

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

function test_constraint_name_unsupported_for_variable_bounds()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    bound_ci = MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    @test !MOI.supports(model, MOI.ConstraintName(), typeof(bound_ci))
    return
end

function test_constraint_name_inverse_lookup_type_filter()
    model = MadNLP.Optimizer()
    x = MOI.add_variable(model)
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0)
    ci_lin = MOI.add_constraint(model, f, MOI.LessThan(1.0))
    MOI.set(model, MOI.ConstraintName(), ci_lin, "shared")
    # Lookup with the matching type returns the CI.
    @test MOI.get(model, typeof(ci_lin), "shared") == ci_lin
    # Lookup with a different CI type returns nothing.
    OtherCI = MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, MOI.LessThan{Float64}}
    @test MOI.get(model, OtherCI, "shared") === nothing
    return
end

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

function test_moimodel_propagates_con_names()
    model = MadNLP.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("max_iter"), 1)
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
    MOI.add_constraint(model, y, MOI.GreaterThan(0.0))
    f_lin = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x), MOI.ScalarAffineTerm(1.0, y)],
        0.0,
    )
    c_lin = MOI.add_constraint(model, f_lin, MOI.LessThan(10.0))
    MOI.set(model, MOI.ConstraintName(), c_lin, "lin")
    f_nl = MOI.ScalarNonlinearFunction(:*, Any[x, y])
    c_nl = MOI.add_constraint(model, f_nl, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ConstraintName(), c_nl, "nl")
    obj = MOI.ScalarNonlinearFunction(:+, Any[
        MOI.ScalarNonlinearFunction(:^, Any[x, 2]),
        MOI.ScalarNonlinearFunction(:^, Any[y, 2]),
    ])
    MOI.set(model, MOI.ObjectiveFunction{typeof(obj)}(), obj)
    MOI.optimize!(model)
    @test model.nlp.con_names == ["lin", "nl"]
    return
end

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
    @test length(labels) == 6
    @test labels[1] == "x"
    @test labels[2] == "y"
    @test labels[3] == "slack[c1]"
    @test labels[4] == "slack[c2]"
    @test labels[5] == "λ[c1]"
    @test labels[6] == "λ[c2]"
    return
end

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
    @test labels == ["x", "y", "zL[x]", "zL[y]", "zU[x]", "zU[y]"]
    return
end

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
    hess = Matrix(solver.kkt.hess_com)
    hess_lbl = MadNLPMOI.hessian_labels(solver)
    ix = findfirst(==("x"), hess_lbl)
    iy = findfirst(==("y"), hess_lbl)
    @test hess[ix, ix] ≈ 2000  atol=1e-6
    @test hess[iy, iy] ≈ 0.02   atol=1e-6
    @test hess[ix, iy] + hess[iy, ix] ≈ 7  atol=1e-6
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test "λ[c_lin1]" in aug_lbl
    @test "λ[c_lin2]" in aug_lbl
    return
end

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
    @test !any(startswith.(aug_lbl, "slack["))
    return
end

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
    @test length(aug_lbl) == 3 + 2 + 3
    @test "slack[c_sum]"   in aug_lbl
    @test "slack[c_bilin]" in aug_lbl
    @test !("slack[c_cubic]" in aug_lbl)
    @test "λ[c_cubic]"     in aug_lbl
    return
end

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

function test_validate_example5_fixed_variable()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.set_attribute(model, "fixed_variable_treatment", MadNLP.MakeParameter)
    JuMP.@variable(model, x == 5, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@variable(model, z, base_name = "z")
    JuMP.@constraint(model, c_sum, x + 2*y + 3*z >= 6)
    JuMP.@objective(model, Min, (x - 1)^2 + (y - 2)^2 + (z - 3)^2)
    JuMP.optimize!(model)
    opt = JuMP.unsafe_backend(model)
    solver = opt.solver
    aug_lbl = MadNLPMOI.kkt_row_labels(solver)
    @test aug_lbl[1] == "y"
    @test aug_lbl[2] == "z"
    @test !("x" in aug_lbl)
    return
end

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
    jac = Matrix(solver.kkt.jac_com)
    @test jac[1, 1] ≈ 13
    @test jac[1, 2] ≈ 17
    @test jac[1, 3] ≈ 19
    @test jac[1, 4] ≈ 23
    return
end

function test_kkt_diagnostic_smoke()
    model = JuMP.Model(MadNLP.Optimizer)
    JuMP.set_attribute(model, "max_iter", 1)
    JuMP.set_attribute(model, "print_level", MadNLP.ERROR)
    JuMP.set_attribute(model, "kkt_system", MadNLP.SparseUnreducedKKTSystem)
    JuMP.@variable(model, 0 <= x <= 10, base_name = "x")
    JuMP.@variable(model, y, base_name = "y")
    JuMP.@constraint(model, c_eq, x + y == 1)
    JuMP.@constraint(model, c_ineq, x * y >= 0.1)
    JuMP.@objective(model, Min, x^2 + 100*y^2)
    JuMP.optimize!(model)
    solver = JuMP.unsafe_backend(model).solver
    diag = MadNLPMOI.kkt_diagnostic(solver)
    @test diag isa MadNLPMOI.KKTDiagnostic
    @test diag.iter == 1
    @test diag.aug_size[1] == diag.aug_size[2]
    @test diag.cond_aug > 0
    @test length(diag.hess_diag) == 3              # 2 vars + 1 slack
    @test length(diag.jac_row_spread) == 2         # 2 contraintes
    @test length(diag.primal) == 3                 # 2 vars + 1 slack
    @test length(diag.dual_con) == 2
    @test length(diag.dual_lb) >= 1                # x has a lower bound
    @test length(diag.dual_ub) >= 1                # x has an upper bound
    @test !isempty(diag.soft_mode)
    # show should not throw
    io = IOBuffer()
    show(io, MIME"text/plain"(), diag)
    @test !isempty(String(take!(io)))
    return
end

end

TestMOIWrapper.runtests()
