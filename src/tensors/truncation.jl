# truncation.jl
#
# Truncation schemes: the abstract type `TruncationScheme`, its concrete
# scheme types and constructors, and the plain-vector truncation `_truncate!`
# helpers, followed by the sector-level truncation machinery that, given a
# per-sector singular value dictionary `Σdata` (sector => vector of singular
# values), computes the truncation dimensions for each scheme.

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

# Sector-level truncation machinery
#-----------------------------------

struct TruncationSpace{S<:ElementarySpace} <: TruncationScheme
    space::S
end
truncspace(space::ElementarySpace) = TruncationSpace(space)

# Compute the total truncation error given truncation dimensions
function _compute_truncerr(Σdata, truncdim, p=2)
    I = keytype(Σdata)
    S = scalartype(valtype(Σdata))
    return _norm((c => view(v, (truncdim[c] + 1):length(v)) for (c, v) in Σdata), p,
                 zero(S))
end

# Compute truncation dimensions
function _compute_truncdim(Σdata, ::NoTruncation, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{Int}(c => length(v) for (c, v) in Σdata)
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationError, p=2)
    I = keytype(Σdata)
    S = real(eltype(valtype(Σdata)))
    truncdim = SectorDict{Int}(c => length(Σc) for (c, Σc) in Σdata)
    truncerr = zero(S)
    while true
        cmin = _findnexttruncvalue(Σdata, truncdim, p)
        isnothing(cmin) && break
        truncdim[cmin] -= 1
        truncerr = _compute_truncerr(Σdata, truncdim, p)
        if truncerr > trunc.ϵ
            truncdim[cmin] += 1
            break
        end
    end
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationDimension, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{Int}(c => length(v) for (c, v) in Σdata)
    while sum(dim(c) * d for (c, d) in truncdim) > trunc.dim
        cmin = _findnexttruncvalue(Σdata, truncdim, p)
        isnothing(cmin) && break
        truncdim[cmin] -= 1
    end
    return truncdim
end
function _compute_truncdim(Σdata, trunc::TruncationSpace, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{Int}(c => min(length(v), dim(trunc.space, c))
                                 for (c, v) in Σdata)
    return truncdim
end

function _compute_truncdim(Σdata, trunc::TruncationCutoff, p=2)
    I = keytype(Σdata)
    truncdim = SectorDict{Int}(c => length(v) for (c, v) in Σdata)
    for (c, v) in Σdata
        newdim = findlast(Base.Fix2(>, trunc.ϵ), v)
        if newdim === nothing
            truncdim[c] = 0
        else
            truncdim[c] = newdim
        end
    end
    for i in 1:(trunc.add_back)
        cmax = _findnextgrowvalue(Σdata, truncdim, p)
        isnothing(cmax) && break
        truncdim[cmax] += 1
    end
    return truncdim
end

# # Combine truncations
# struct MultipleTruncation{T<:Tuple{Vararg{TruncationScheme}}} <: TruncationScheme
#     truncations::T
# end
# function Base.:&(a::MultipleTruncation, b::MultipleTruncation)
#     return MultipleTruncation((a.truncations..., b.truncations...))
# end
# function Base.:&(a::MultipleTruncation, b::TruncationScheme)
#     return MultipleTruncation((a.truncations..., b))
# end
# function Base.:&(a::TruncationScheme, b::MultipleTruncation)
#     return MultipleTruncation((a, b.truncations...))
# end
# Base.:&(a::TruncationScheme, b::TruncationScheme) = MultipleTruncation((a, b))

# function _compute_truncdim(Σdata, trunc::MultipleTruncation, p::Real=2)
#     truncdim = _compute_truncdim(Σdata, trunc.truncations[1], p)
#     for k in 2:length(trunc.truncations)
#         truncdimₖ = _compute_truncdim(Σdata, trunc.truncations[k], p)
#         for (c, d) in truncdim
#             truncdim[c] = min(d, truncdimₖ[c])
#         end
#     end
#     return truncdim
# end

# auxiliary function
function _findnexttruncvalue(Σdata, truncdim::SectorDict{Int}, p::Real)
    # early return
    (isempty(Σdata) || all(iszero, values(truncdim))) && return nothing

    # find some suitable starting candidate
    cmin = findfirst(>(0), truncdim)
    σmin = Σdata[cmin][truncdim[cmin]]

    # find the actual minimum singular value
    for (c, σs) in Σdata
        if truncdim[c] > 0
            σ = σs[truncdim[c]]
            if σ < σmin
                cmin, σmin = c, σ
            end
        end
    end
    return cmin
end
function _findnextgrowvalue(Σdata, truncdim::SectorDict{Int}, p::Real)
    istruncated = SectorDict{Bool}(c => (d < length(Σdata[c])) for (c, d) in truncdim)
    # early return
    (isempty(Σdata) || !any(values(istruncated))) && return nothing

    # find some suitable starting candidate
    cmax = findfirst(istruncated)
    σmax = Σdata[cmax][truncdim[cmax] + 1]

    # find the actual maximal singular value
    for (c, σs) in Σdata
        if istruncated[c]
            σ = σs[truncdim[c] + 1]
            if σ > σmax
                cmax, σmax = c, σ
            end
        end
    end
    return cmax
end

function _compute_truncdim(Σdata, trunc::TruncateDimCutoff, p = 2)
    n = _norm(Σdata, p, 0.0)
    truncdim1 = _compute_truncdim(Σdata, truncbelow(trunc.ϵ * n, trunc.add_back), p)
    if compute_size(truncdim1) <= trunc.D
        return truncdim1
    end
    return _compute_truncdim(Σdata, truncdim(trunc.D), p)
end

function _compute_truncdim(Σdata, trunc::TruncateRelError, p = 2)
    n = _norm(Σdata, p, 0.0)
    return _compute_truncdim(Σdata, truncbelow(trunc.ϵ * n, trunc.add_back), p)
end
