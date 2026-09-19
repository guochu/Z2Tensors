# Plain-array tensor factorization utilities, ported from TEMPO's
# src/tensorops/tensorfactorizations.jl (matrix-level `_truncate!` helpers are
# adapted from TEMPO's src/tensorops/truncation.jl) and aligned with this
# package's truncation schemes and MatrixAlgebraKit backends.
# Also defines the truncation scheme types themselves; the sector-level
# truncation machinery (`_compute_truncdim`) lives in tensors/truncation.jl.
# Plain-array tensor operations (permute, tie, isometry, kron) live in
# auxiliary/tensoroperations.jl.

# Truncation schemes
#--------------------
abstract type TruncationScheme end

struct NoTruncation <: TruncationScheme
end
notrunc() = NoTruncation()

struct TruncationError{T<:Real} <: TruncationScheme
    ϵ::T
end
truncerr(epsilon::Real) = TruncationError(epsilon)

struct TruncationDimension <: TruncationScheme
    dim::Int
end
truncdim(d::Int) = TruncationDimension(d)
truncdim(; D::Int) = truncdim(D)

struct TruncationCutoff{T<:Real} <: TruncationScheme
    ϵ::T
    add_back::Int
end
truncbelow(epsilon::Real, add_back::Int = 0) = TruncationCutoff(epsilon, add_back)

"""
    struct TruncateDimCutoff
first normalize the singular value spectrum by its p-norm, then truncate singular
values below the relative cutoff ϵ; if the remaining bond dimension is larger than D,
truncate it below D. The largest add_back singular values are kept even if they fall
below the cutoff.
Return the p-norm of the truncated singular values.
"""
struct TruncateDimCutoff <: TruncationScheme
    D::Int
    ϵ::Float64
    add_back::Int
    function TruncateDimCutoff(D::Int, ϵ::Real, add_back::Int)
        add_back <= D ||
            throw(ArgumentError("add_back (= $add_back) cannot be larger than D (= $D)"))
        return new(D, ϵ, add_back)
    end
end
TruncateDimCutoff(; D::Int, ϵ::Real, add_back::Int = 0) =
    TruncateDimCutoff(D, convert(Float64, ϵ), add_back)
truncdimcutoff(D::Int, epsilon::Real; add_back::Int = 0) =
    TruncateDimCutoff(D, epsilon, add_back)
truncdimcutoff(; D::Int, ϵ::Real, add_back::Int = 0) =
    TruncateDimCutoff(D, convert(Float64, ϵ), add_back)

"""
    struct TruncateRelError
truncate singular values below a relative cutoff ϵ, i.e. the singular value vector is
first normalized (using its p-norm) and singular values below ϵ are discarded; if fewer
than `add_back` singular values remain, keep `add_back` of them.
"""
struct TruncateRelError <: TruncationScheme
    ϵ::Float64
    add_back::Int
end
TruncateRelError(; ϵ::Real, add_back::Int = 0) =
    TruncateRelError(convert(Float64, ϵ), add_back)
truncrelerr(epsilon::Real, add_back::Int = 0) =
    TruncateRelError(convert(Float64, epsilon), add_back)
truncrelerr(; ϵ::Real, add_back::Int = 0) =
    TruncateRelError(convert(Float64, ϵ), add_back)

compute_size(v::AbstractVector) = length(v)
function compute_size(v::AbstractDict)
    init = 0
    for (c, b) in v
        init += dim(c) * b
    end
    return init
end

# Matrix-level truncation of a singular value vector (descending order)
#-----------------------------------------------------------------------
_truncate!(v::AbstractVector{<:Real}, ::NoTruncation, p::Real = 2) = v, 0.0

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncationDimension, p::Real = 2)
    dtrunc = min(length(v), trunc.dim)
    truncerr = norm(view(v, (dtrunc + 1):length(v)), p)
    resize!(v, dtrunc)
    return v, truncerr
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncationError, p::Real = 2)
    dtrunc = length(v)
    while dtrunc > 0 && norm(view(v, dtrunc:length(v)), p) <= trunc.ϵ
        dtrunc -= 1
    end
    return _truncate!(v, TruncationDimension(dtrunc), p)
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncationCutoff, p::Real = 2)
    dtrunc = findlast(Base.Fix2(>, trunc.ϵ), v)
    dtrunc = isnothing(dtrunc) ? 0 : dtrunc
    dtrunc = max(dtrunc, trunc.add_back) # keep at least add_back singular values
    return _truncate!(v, TruncationDimension(dtrunc), p)
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncateRelError, p::Real = 2)
    sca = norm(v, p)
    dtrunc = findlast(Base.Fix2(>, sca * trunc.ϵ), v)
    dtrunc = isnothing(dtrunc) ? 0 : dtrunc
    dtrunc = max(dtrunc, trunc.add_back) # keep at least add_back singular values
    return _truncate!(v, TruncationDimension(dtrunc), p)
end

function _truncate!(v::AbstractVector{<:Real}, trunc::TruncateDimCutoff, p::Real = 2)
    sca = norm(v, p)
    dtrunc = findlast(Base.Fix2(>, sca * trunc.ϵ), v)
    dtrunc = isnothing(dtrunc) ? 0 : dtrunc
    dtrunc = max(dtrunc, trunc.add_back)   # keep at least add_back singular values
    dtrunc = min(dtrunc, trunc.D)          # but never more than D
    v, err = _truncate!(v, TruncationDimension(dtrunc), p)
    return v, sca == zero(sca) ? err : err / sca
end

# Truncated SVD of plain matrices and dense tensors
#---------------------------------------------------
"""
    tsvd!(a; trunc=NoTruncation(), alg=SDD())

Perform a truncated singular value decomposition of the matrix `a`, returning `(u, s, v, err)`,
where `err` is the truncation error (2-norm of the discarded singular values), `trunc` specifies the truncation scheme,
and `alg` selects the SVD driver: `SDD()` (divide-and-conquer with safe fallback, LAPACK `gesdvd`) or `SVD()` (QR iteration, LAPACK `gesvd`).

Note: the input matrix `a` is used as workspace and may be **destroyed/overwritten** in place; pass a copy if the input must be preserved.
"""
function tsvd!(a::StridedMatrix; trunc::TruncationScheme = NoTruncation(),
               alg::Union{SVD,SDD} = SDD())
    u, S, v = MatrixAlgebraKit.svd_compact!(a;
                                           alg = alg isa SDD ? SafeDivideAndConquer() :
                                                 QRIteration())
    s = S.diag
    d_old = length(s)
    s, err = _truncate!(s, trunc)
    d = length(s)
    if d == d_old
        return u, s, v, err
    else
        return u[:, 1:d], s, v[1:d, :], err
    end
end

"""
    tsvd(a; trunc=NoTruncation(), alg=SDD())

Non-mutating version of `tsvd!(a; trunc, alg)`: the input matrix is copied before decomposition.
"""
tsvd(a::AbstractMatrix; kwargs...) = tsvd!(copy(a); kwargs...)

"""
    tsvd!(a, left, right; trunc=NoTruncation(), alg=SDD())

Perform a truncated singular value decomposition of the tensor `a` with dimensions grouped into `left`/`right`,
returning `(u, s, v, err)`, where `u` and `v` are the left and right singular tensors.

Note: the input tensor `a` itself is not modified (the permuted matrix is copied internally, since the
decomposition requires a contiguous workspace); the `!` marks the variant that reuses that copy as workspace.
"""
function tsvd!(a::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
               trunc::TruncationScheme = NoTruncation(),
               alg::Union{SVD,SDD} = SDD()) where {T<:Number,N,N1,N2}
    b, ushape, vshape = _tomat(a, left, right)
    isa(b, StridedMatrix) || (b = copy(b))
    u, s, v, err = tsvd!(b; trunc = trunc, alg = alg)
    md = length(s)
    return reshape(u, (ushape..., md)), s, reshape(v, (md, vshape...)), err
end

"""
    tsvd(a, left, right; trunc=NoTruncation(), alg=SDD())

Non-mutating version of `tsvd!(a, left, right; trunc, alg)`: the input tensor is copied before decomposition.
"""
function tsvd(a::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int};
              kwargs...) where {N1,N2}
    return tsvd!(copy(a), left, right; kwargs...)
end

# Orthogonal factorizations of plain matrices and dense tensors
#---------------------------------------------------------------
"""
    leftorth!(A; alg=QRpos(), atol=0)

Keyword version of `leftorth!` acting on plain matrices, equivalent to `leftorth!(A, alg, atol)`.
"""
function leftorth!(A::StridedMatrix;
                   alg::Union{QR,QRpos,SVD,SDD,Polar} = QRpos(),
                   atol::Real = zero(float(real(scalartype(A)))))
    return leftorth!(A, alg, atol)
end
"""
    leftorth!(A, left, right; alg=QRpos(), atol=0)

Left-orthogonalize the tensor `A` with dimensions grouped into `left`/`right`, returning `(u, v)`,
where `u` has dimensions `(left..., s)` and `v` has dimensions `(s, right...)`.

Note: the input tensor `A` itself is not modified (the permuted matrix is copied internally, since the
factorization requires a contiguous workspace); the `!` marks the variant that reuses that copy as workspace.
"""
function leftorth!(A::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                   alg::Union{QR,QRpos,SVD,SDD,Polar} = QRpos(),
                   atol::Real = zero(float(real(scalartype(A))))) where {T,N,N1,N2}
    A2, dimu, dimv = _tomat(A, left, right)
    isa(A2, StridedMatrix) || (A2 = copy(A2))
    u, v = leftorth!(A2, alg, atol)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    leftorth(A; alg=QRpos(), atol=0)
    leftorth(A, left, right; alg=QRpos(), atol=0)

Non-mutating version of `leftorth!`: the input is copied before orthogonalization.
"""
leftorth(A::AbstractMatrix; kwargs...) = leftorth!(copy(A); kwargs...)
function leftorth(A::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                  kwargs...) where {N1,N2}
    return leftorth!(copy(A), left, right; kwargs...)
end

"""
    rightorth!(A; alg=LQpos(), atol=0)

Keyword version of `rightorth!` acting on plain matrices, equivalent to `rightorth!(A, alg, atol)`.
"""
function rightorth!(A::StridedMatrix;
                    alg::Union{LQ,LQpos,SVD,SDD,Polar} = LQpos(),
                    atol::Real = zero(float(real(scalartype(A)))))
    return rightorth!(A, alg, atol)
end
"""
    rightorth!(A, left, right; alg=LQpos(), atol=0)

Right-orthogonalize the tensor `A` with dimensions grouped into `left`/`right`, returning `(u, v)`,
where `u` has dimensions `(left..., s)` and `v` has dimensions `(s, right...)`.

Note: the input tensor `A` itself is not modified (the permuted matrix is copied internally, since the
factorization requires a contiguous workspace); the `!` marks the variant that reuses that copy as workspace.
"""
function rightorth!(A::AbstractArray{T,N}, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                    alg::Union{LQ,LQpos,SVD,SDD,Polar} = LQpos(),
                    atol::Real = zero(float(real(scalartype(A))))) where {T,N,N1,N2}
    A2, dimu, dimv = _tomat(A, left, right)
    isa(A2, StridedMatrix) || (A2 = copy(A2))
    u, v = rightorth!(A2, alg, atol)
    s = size(v, 1)
    return reshape(u, dimu..., s), reshape(v, s, dimv...)
end

"""
    rightorth(A; alg=LQpos(), atol=0)
    rightorth(A, left, right; alg=LQpos(), atol=0)

Non-mutating version of `rightorth!`: the input is copied before orthogonalization.
"""
rightorth(A::AbstractMatrix; kwargs...) = rightorth!(copy(A); kwargs...)
function rightorth(A::AbstractArray, left::NTuple{N1,Int}, right::NTuple{N2,Int};
                   kwargs...) where {N1,N2}
    return rightorth!(copy(A), left, right; kwargs...)
end

function _tomat(a::AbstractArray{T,N}, left::NTuple{N1,Int},
                right::NTuple{N2,Int}) where {T<:Number,N,N1,N2}
    (N == N1 + N2) || throw(DimensionMismatch())
    newindex = (left..., right...)
    a1 = permute(a, newindex)
    shape_a = size(a1)
    dimu = ntuple(i -> shape_a[i], N1)
    s1 = prod(dimu)
    dimv = ntuple(i -> shape_a[N1 + i], N2)
    s2 = prod(dimv)
    return reshape(a1, s1, s2), dimu, dimv
end

# Entropy of a normalized singular value spectrum
#-------------------------------------------------
"""
    renyi_entropy(v::AbstractVector{<:Real}; α::Real=1)

Compute the Rényi entropy of the vector `v`. For `α=1` this reduces to the Shannon entropy `-Σᵢ vᵢ log(vᵢ)`,
and otherwise it is `1/(1-α) * log(Σᵢ vᵢ^α)`.
The entries of `v` must be nonnegative and sum to 1 (e.g. normalized squared singular values).
"""
function renyi_entropy(v::AbstractVector{<:Real}; α::Real = 1)
    α = convert(eltype(v), α)
    a = _check_and_filter(v)
    if α == one(α)
        return -dot(a, log.(a))
    else
        a = a .^ (α)
        return (1 / (1 - α)) * log(sum(a))
    end
end

function _check_and_filter(v::AbstractVector{<:Real}; tol::Real = 1.0e-12)
    (abs(sum(v) - 1) <= tol) ||
        throw(ArgumentError("sum of singular values not equal to 1"))
    oo = zero(eltype(v))
    for item in v
        ((item < oo) && (-item > tol)) &&
            throw(ArgumentError("negative singular values"))
    end
    return [item for item in v if abs(item) > tol]
end
