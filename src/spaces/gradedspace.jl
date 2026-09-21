
struct Z2Space <: ElementarySpace
    dims::NTuple{2,Int}
    dual::Bool
end

sectortype(::Type{<:Z2Space}) = Z2Irrep

function Z2Space(dims; dual::Bool=false)
    N = 2
    d = ntuple(n -> 0, N)
    isset = ntuple(n -> false, N)
    for (c, dc) in dims
        k = convert(Z2Irrep, c)
        i = k.n + 1
        dc < 0 && throw(ArgumentError("Sector $k has negative dimension $dc"))
        isset[i] && throw(ArgumentError("Sector $c appears multiple times"))
        isset = TupleTools.setindex(isset, true, i)
        d = TupleTools.setindex(d, dc, i)
    end
    return Z2Space(d, dual)
end
function Z2Space(dims::Pair; dual::Bool=false)
    return Z2Space((dims,); dual=dual)
end
Z2Space(d1::Pair, d2::Pair, rdims::Pair...; dual::Bool=false) =
    Z2Space((d1, d2, rdims...); dual=dual)
Z2Space(dims::AbstractDict; dual::Bool=false) = Z2Space(dims...; dual=dual)
Z2Space(g::Base.Generator; dual::Bool=false) = Z2Space(g...; dual=dual)
Z2Space(dims::NTuple{2,Int}; dual::Bool=false) = Z2Space(dims, dual)
Z2Space(d1::Int, d2::Int; dual::Bool=false) = Z2Space((d1, d2), dual)
Z2Space(; dual::Bool=false) = Z2Space((0, 0), dual)

Base.hash(V::Z2Space, h::UInt) = hash(V.dual, hash(V.dims, h))

function dim(V::Z2Space)
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V); init=0)
end
dim(V::Z2Space, c::Z2Irrep) = V.dims[c.n+1]

sectors(V::Z2Space) = [Z2Irrep(i-1) for i in 1:2 if V.dims[i] != 0]

hassector(V::Z2Space, s::Z2Irrep) = dim(V, s) != 0

Base.conj(V::Z2Space) = Z2Space(V.dims, !V.dual)
isdual(V::Z2Space) = V.dual

# equality / comparison
function Base.:(==)(V₁::Z2Space, V₂::Z2Space)
    return (V₁.dims == V₂.dims) && V₁.dual == V₂.dual
end

Base.oneunit(S::Type{Z2Space}) = S(one(Z2Irrep) => 1)
Base.zero(S::Type{Z2Space}) = S(one(Z2Irrep) => 0)
Base.oneunit(V::Z2Space) = oneunit(typeof(V))
Base.zero(V::Z2Space) = zero(typeof(V))

# space consisting of a single trivial charge, isomorphic to the underlying field
isunitspace(V::Z2Space) = (V.dims[1] == 1) && (V.dims[2] == 0)

function ⊕(V₁::Z2Space, V₂::Z2Space)
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{Int}()
    for c in union(sectors(V₁), sectors(V₂))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V₁, c) + dim(V₂, c)
    end
    return Z2Space(dims; dual=dual1)
end
⊕(V₁::Z2Space, V₂::Z2Space, Vs::Z2Space...) = ⊕(⊕(V₁, V₂), Vs...)

function fuse(V₁::Z2Space, V₂::Z2Space)
    dims = SectorDict{Int}()
    for a in sectors(V₁), b in sectors(V₂)
        c = a ⊗ b
        dims[c] = get(dims, c, 0) + Nsymbol(a, b, c) * dim(V₁, a) * dim(V₂, b)
    end
    return Z2Space(dims)
end
function fuse(V₁::Z2Space, V₂::Z2Space, V₃::Z2Space...)
    return fuse(fuse(V₁, V₂), V₃...)
end
fuse(V::Z2Space) = V
fuse(v::NTuple{N, Z2Space}) where {N} = N == 0 ? oneunit(Z2Space) : fuse(v...)

function infimum(V₁::Z2Space, V₂::Z2Space)
    if V₁.dual == V₂.dual
        Z2Space(c => min(dim(V₁, c), dim(V₂, c))
                for c in union(sectors(V₁), sectors(V₂)); dual=V₁.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V₁::Z2Space, V₂::Z2Space)
    if V₁.dual == V₂.dual
        Z2Space(c => max(dim(V₁, c), dim(V₂, c))
                for c in union(sectors(V₁), sectors(V₂)); dual=V₁.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end
