# Simple reference to getting and setting BLAS threads
#------------------------------------------------------
set_num_blas_threads(n::Integer) = LinearAlgebra.BLAS.set_num_threads(n)
get_num_blas_threads() = LinearAlgebra.BLAS.get_num_threads()

# Factorization algorithms
#--------------------------
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct SDD <: OrthogonalFactorizationAlgorithm # lapack's default divide and conquer algorithm
end
struct SVD <: OrthogonalFactorizationAlgorithm
end
struct Polar <: OrthogonalFactorizationAlgorithm
end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()

Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm
const SVDAlg = Union{SVD,SDD}

# TODO: define for CuMatrix if we support this
function one!(A::StridedMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

# Matrix level factorizations, implemented via MatrixAlgebraKit
#---------------------------------------------------------------
using MatrixAlgebraKit: MatrixAlgebraKit, trunctol,
                        LAPACK_QRIteration, LAPACK_DivideAndConquer

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MatrixAlgebraKit.left_orth!(A; alg = :qr, positive = (alg isa QRpos))
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
    return MatrixAlgebraKit.left_orth!(A; alg = :svd, trunc = trunctol(atol = atol))
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MatrixAlgebraKit.left_orth!(A; alg = :polar)
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MatrixAlgebraKit.right_orth!(A; alg = :lq, positive = (alg isa LQpos))
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
    return MatrixAlgebraKit.right_orth!(A; alg = :svd, trunc = trunctol(atol = atol))
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    return MatrixAlgebraKit.right_orth!(A; alg = :polar)
end

function _svd!(A::StridedMatrix{T}, alg::Union{SVD,SDD}) where {T<:BlasFloat}
    svdalg = alg isa SVD ? LAPACK_QRIteration() : LAPACK_DivideAndConquer()
    U, S, Vᴴ = MatrixAlgebraKit.svd_compact!(A; alg = svdalg)
    return U, S.diag, Vᴴ
end
