testset = [
    # Temporarily commented out since LapackGPUSolver does not currently support sparse callbacks
    [
        "LapackGPU-CUSOLVERRF",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-CUSOLVERRF",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.RFSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-CUSOLVERRF",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.CuCholeskySolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-CUSOLVERRF",
        ()->MadNLP.Optimizer(
            linear_solver=MadNLPGPU.GLUSolver,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-BUNCHKAUFMAN",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.BUNCHKAUFMAN,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-LU",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.LU,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-QR",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.QR,
            print_level=MadNLP.ERROR
        ),
        [],
    ],
    [
        "LapackGPU-CHOLESKY",
        ()->MadNLP.Optimizer(
            linear_solver=LapackGPUSolver,
            lapack_algorithm=MadNLP.CHOLESKY,
            print_level=MadNLP.ERROR
        ),
        ["infeasible", "lootsma", "eigmina"], # KKT system not PD
    ],
]

@testset "MadNLPGPU test" begin
    MadNLPTests.test_linear_solver(LapackGPUSolver,Float32)
    MadNLPTests.test_linear_solver(LapackGPUSolver,Float64)
    # Test LapackGPU wrapper
    for (name,optimizer_constructor,exclude) in testset
        test_madnlp(name,optimizer_constructor,exclude; Arr= CuArray)
    end
end
