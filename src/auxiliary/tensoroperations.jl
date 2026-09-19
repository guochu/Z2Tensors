# Plain-array tensor operations: permute views, tie/group axes, isometries and
# Kronecker products. The factorization-related utilities (truncation schemes,
# tsvd, orthogonal factorizations) live in auxiliary/tensorfactorizations.jl.

# Array helpers
#---------------
"""
    permute(m::AbstractArray, perm)

Return a view of `m` with its dimensions permuted according to `perm` (a `PermutedDimsArray`), without copying data.
"""
permute(m::AbstractArray, perm) = PermutedDimsArray(m, perm)
"""
    permute(m::AbstractArray, left, right)

Group the dimensions into `left` and `right` and place them in that order, equivalent to `permute(m, (left..., right...))`.
"""
permute(m::AbstractArray, left, right) = permute(m, (left..., right...))

function isometry(::Type{T}, m::Int, n::Int) where {T<:Number}
    r = zeros(T, m, n)
    for i in 1:min(m, n)
        r[i, i] = 1
    end
    return r
end
"""
    isometry(T, m, n)
    isometry(T, d)
    isometry(m, n)
    isometry(d)

Return the rectangular identity (isometry) matrix of size `m × n` (`d × d`) with element type `T` (default `Float64`).
"""
isometry(::Type{T}, d::Int) where {T<:Number} = isometry(T, d, d)
isometry(m::Int, n::Int) = isometry(Float64, m, n)
isometry(d::Int) = isometry(Float64, d)

function _group_extent(extent::NTuple{N,Int}, idx::NTuple{N1,Int}) where {N,N1}
    return ntuple(Val(N1)) do i
        l = sum(idx[j] for j in 1:(i - 1); init = 0)
        prod(ntuple(j -> extent[l + j], idx[i]))
    end
end

"""
    tie(a::AbstractArray, axs)

Reshape the tensor `a` by grouping consecutive axes according to `axs`, where `sum(axs)` must equal `ndims(a)`.
"""
function tie(a::AbstractArray{T,N}, axs::NTuple{N1,Int}) where {T,N,N1}
    (sum(axs) != N) && error("total number of axes should equal to tensor rank")
    return reshape(a, _group_extent(size(a), axs))
end

# Kronecker product of plain arrays
#-----------------------------------
"""
    Base.kron(a::AbstractArray{<:Number,N}, b::AbstractArray{<:Number,N})

Kronecker product of two arrays of equal rank `N`: the output size along axis `j`
is `size(a, j) * size(b, j)` and
`c[(iⱼ - 1) * size(b, j) + ibⱼ, ...] = a[i₁, i₂, ...] * b[ib₁, ib₂, ...]`.
This generalizes `Base.kron` beyond vectors and matrices; for those ranks the
result is unchanged.
"""
function Base.kron(a::AbstractArray{Ta,N}, b::AbstractArray{Tb,N}) where {Ta<:Number,
                                                                          Tb<:Number,N}
    N == 0 && error("empty tensors")
    sa = size(a)
    sb = size(b)
    sc = ntuple(i -> sa[i] * sb[i], Val(N))
    c = Array{promote_type(Ta, Tb), N}(undef, sc)
    for index in CartesianIndices(a)
        ranges = ntuple(j -> ((index[j] - 1) * sb[j] + 1):(index[j] * sb[j]), Val(N))
        c[ranges...] = a[index] * b
    end
    return c
end
