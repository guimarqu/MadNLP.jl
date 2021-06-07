module LapackGPU

import ..MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    CUBLAS, CUSOLVER, CuVector, CuMatrix,
    AbstractOptions, AbstractLinearSolver, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, LapackCPU, tril_to_full!

import .CUSOLVER: libcusolver, cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t, dense_handle
import CUDA: R_64F

const INPUT_MATRIX_TYPE = :dense

@enum(Algorithms::Int, BUNCHKAUFMAN = 1, LU = 2, QR = 3)
@kwdef mutable struct Options <: AbstractOptions
    lapackgpu_algorithm::Algorithms = BUNCHKAUFMAN
end

mutable struct Solver <: AbstractLinearSolver
    dense::Matrix{Float64}
    fact::CuMatrix{Float64}
    rhs::CuVector{Float64}
    work::CuVector{Float64}
    lwork
    work_host::Vector{Float64}
    lwork_host
    info::CuVector{Int32}
    etc::Dict{Symbol,Any} # throw some algorithm-specific things here
    opt::Options
    logger::Logger
end

function Solver(dense::Matrix{Float64};
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                opt=Options(),logger=Logger(),
                kwargs...)

    set_options!(opt,option_dict,kwargs...)
    fact = CuMatrix{Float64}(undef,size(dense))
    rhs = CuVector{Float64}(undef,size(dense,1))
    work  = CuVector{Float64}(undef, 1)
    lwork = Int32[1]
    work_host  = Vector{Float64}(undef, 1)
    lwork_host = Int32[1]
    info = CuVector{Int32}(undef,1)
    etc = Dict{Symbol,Any}()

    return Solver(dense,fact,rhs,work,lwork,work_host,lwork_host,info,etc,opt,logger)
end

function factorize!(M::Solver)
    if M.opt.lapackgpu_algorithm == BUNCHKAUFMAN
        factorize_bunchkaufman!(M)
    elseif M.opt.lapackgpu_algorithm == LU
        factorize_lu!(M)
    elseif M.opt.lapackgpu_algorithm == QR
        factorize_qr!(M)
    else
        error(LOGGER,"Invalid lapackgpu_algorithm")
    end
end
function solve!(M::Solver,x)
    if M.opt.lapackgpu_algorithm == BUNCHKAUFMAN
        solve_bunchkaufman!(M,x)
    elseif M.opt.lapackgpu_algorithm == LU
        solve_lu!(M,x)
    elseif M.opt.lapackgpu_algorithm == QR
        solve_qr!(M,x)
    else
        error(LOGGER,"Invalid lapackgpu_algorithm")
    end
end

is_inertia(M::Solver) = false # TODO: implement inertia(M::Solver) for BUNCHKAUFMAN 
improve!(M::Solver) = false
introduce(M::Solver) = "Lapack-GPU ($(M.opt.lapackgpu_algorithm))"

function factorize_bunchkaufman!(M::Solver)
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef,size(M.dense,1)))    
    haskey(M.etc,:ipiv64) || (M.etc[:ipiv64] = CuVector{Int64}(undef,length(M.etc[:ipiv]))) 

    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDsytrf_bufferSize(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDsytrf(
        CUSOLVER.dense_handle(),CUBLAS.CUBLAS_FILL_MODE_LOWER,
        Int32(size(M.fact,1)),M.fact,Int32(size(M.fact,2)),
        M.etc[:ipiv],M.work,M.lwork[],M.info)

    # # need to send the factorization back to cpu to call mkl sytrs --------------
    # haskey(M.etc,:fact_cpu) || (M.etc[:fact_cpu] = Matrix{Float64}(undef,size(M.dense)))
    # haskey(M.etc,:ipiv_cpu) || (M.etc[:ipiv_cpu] = Vector{Int}(undef,size(M.dense,1)))
    # copyto!(M.etc[:fact_cpu],M.fact)
    # copyto!(M.etc[:ipiv_cpu],M.etc[:ipiv])
    # # ---------------------------------------------------------------------------
    return M
end

function solve_bunchkaufman!(M::Solver,x)
    
    copyto!(M.etc[:ipiv64],M.etc[:ipiv])
    copyto!(M.rhs,x)
    ccall((:cusolverDnXsytrs_bufferSize, libcusolver()), cusolverStatus_t,
          (CUSOLVER.cusolverDnHandle_t, cublasFillMode_t, Int64, Int64, cudaDataType,
           CuPtr{Cdouble}, Int64, CuPtr{Int64}, cudaDataType,
           CuPtr{Cdouble}, Int64, Ptr{Int64}, Ptr{Int64}),
          dense_handle(), CUBLAS.CUBLAS_FILL_MODE_LOWER,
          size(M.fact,1),1,R_64F,M.fact,size(M.fact,2),
          M.etc[:ipiv64],R_64F,M.rhs,length(M.rhs),M.lwork,M.lwork_host)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    length(M.work_host) < M.lwork_host[] && resize!(work_host,Int(M.lwork_host[]))
    ccall((:cusolverDnXsytrs, libcusolver()), cusolverStatus_t,
          (cusolverDnHandle_t, cublasFillMode_t, Int64, Int64, cudaDataType,
           CuPtr{Cdouble}, Int64, CuPtr{Int64}, cudaDataType,
           CuPtr{Cdouble}, Int64, CuPtr{Cdouble}, Int64, Ptr{Cdouble}, Int64,
           CuPtr{Int64}),
          dense_handle(),CUBLAS.CUBLAS_FILL_MODE_LOWER,
          size(M.fact,1),1,R_64F,M.fact,size(M.fact,2),
          M.etc[:ipiv64],R_64F,M.rhs,length(M.rhs),M.work,M.lwork[],M.work_host,M.lwork_host[],M.info)
    copyto!(x,M.rhs)
    
    return x
end

function factorize_lu!(M::Solver)
    haskey(M.etc,:ipiv) || (M.etc[:ipiv] = CuVector{Int32}(undef,size(M.dense,1)))
    tril_to_full!(M.dense)
    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDgetrf_bufferSize(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),
        M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDgetrf(
        CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),
        M.fact,Int32(size(M.fact,2)),M.work,M.etc[:ipiv],M.info)
    return M
end

function solve_lu!(M::Solver,x)
    copyto!(M.rhs,x)
    CUSOLVER.cusolverDnDgetrs(
        CUSOLVER.dense_handle(),CUBLAS.CUBLAS_OP_N,
        Int32(size(M.fact,1)),Int32(1),M.fact,Int32(size(M.fact,2)),
        M.etc[:ipiv],M.rhs,Int32(length(M.rhs)),M.info)
    copyto!(x,M.rhs)
    return x
end

function factorize_qr!(M::Solver)
    haskey(M.etc,:tau) || (M.etc[:tau] = CuVector{Float64}(undef,size(M.dense,1)))
    haskey(M.etc,:one) || (M.etc[:one] = ones(1))
    tril_to_full!(M.dense)
    copyto!(M.fact,M.dense)
    CUSOLVER.cusolverDnDgeqrf_bufferSize(CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),M.fact,Int32(size(M.fact,2)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDgeqrf(CUSOLVER.dense_handle(),Int32(size(M.fact,1)),Int32(size(M.fact,2)),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.work,M.lwork[],M.info)
    return M
end

function solve_qr!(M::Solver,x)
    copyto!(M.rhs,x)
    CUSOLVER.cusolverDnDormqr_bufferSize(CUSOLVER.dense_handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_OP_T,
                                         Int32(size(M.fact,1)),Int32(1),Int32(length(M.etc[:tau])),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.rhs,Int32(length(M.rhs)),M.lwork)
    length(M.work) < M.lwork[] && resize!(M.work,Int(M.lwork[]))
    CUSOLVER.cusolverDnDormqr(CUSOLVER.dense_handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_OP_T,
                              Int32(size(M.fact,1)),Int32(1),Int32(length(M.etc[:tau])),M.fact,Int32(size(M.fact,2)),M.etc[:tau],M.rhs,Int32(length(M.rhs)),M.work,M.lwork[],M.info)
    CUBLAS.cublasDtrsm_v2(CUBLAS.handle(),CUBLAS.CUBLAS_SIDE_LEFT,CUBLAS.CUBLAS_FILL_MODE_UPPER,CUBLAS.CUBLAS_OP_N,CUBLAS.CUBLAS_DIAG_NON_UNIT,
                          Int32(size(M.fact,1)),Int32(1),M.etc[:one],M.fact,Int32(size(M.fact,2)),M.rhs,Int32(length(M.rhs)))
    copyto!(x,M.rhs)
    return x
end

end # module



# forgiving names
lapackgpu = LapackGPU
LAPACKGPU = LapackGPU
