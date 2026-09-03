
struct ProductSpace{N} <: CompositeSpace{Z2Space}
    spaces::NTuple{N,Z2Space}
    ProductSpace{N}(spaces::NTuple{N,Z2Space}) where {N} = new{N}(spaces)
end

function ProductSpace(spaces::Tuple{Vararg{Z2Space}})
    return ProductSpace{length(spaces)}(spaces)
end
function ProductSpace(space1::Z2Space, rspaces::Vararg{Z2Space})
    return ProductSpace((space1, rspaces...))
end

ProductSpace(P::ProductSpace) = P

# Corresponding methods
#-----------------------
dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = prod(dims(P))

dual(P::ProductSpace{0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# more specific methods

sectors(P::ProductSpace) = _sectors(P, sectortype(P))
function _sectors(P::ProductSpace{N}, ::Type{<:Sector}) where {N}
    return product(map(sectors, P.spaces)...)
end


function hassector(V::ProductSpace{N}, s::NTuple{N}) where {N}
    return reduce(&, map(hassector, V.spaces, s); init=true)
end


function dims(P::ProductSpace{N}, sector::NTuple{N,<:Sector}) where {N}
    return map(dim, P.spaces, sector)
end

function dim(P::ProductSpace{N}, sector::NTuple{N,<:Sector}) where {N}
    return reduce(*, dims(P, sector); init=1)
end

function blocksectors(P::ProductSpace{N}) where {N}
    I = Z2Irrep
    bs = Vector{I}()
    if N == 0
        push!(bs, one(I))
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            c = ⊗(s...)
            if !(c in bs)
                push!(bs, c)
            end
        end
    end
    return sort!(bs)
end


function fusiontrees(P::ProductSpace{N}, blocksector::Sector) where {N}
    blocksector isa Z2Irrep || throw(SectorMismatch())
    uncoupled = map(sectors, P.spaces)
    return fusiontrees(uncoupled, blocksector)
end

hasblock(P::ProductSpace, c::Sector) = !isempty(fusiontrees(P, c))

function blockdim(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    d = 0
    for f in fusiontrees(P, c)
        d += dim(P, f.uncoupled)
    end
    return d
end

function Base.:(==)(P1::ProductSpace{N}, P2::ProductSpace{N}) where {N}
    return (P1.spaces == P2.spaces)
end
Base.:(==)(P1::ProductSpace, P2::ProductSpace) = false

Base.hash(P::ProductSpace, h::UInt) = hash(P.spaces, hash(Z2Space, h))

# Default construction from product of spaces
#---------------------------------------------
⊗(V::Z2Space, Vrest::Z2Space...) = ProductSpace(V, Vrest...)
⊗(P::ProductSpace) = P
⊗(P1::ProductSpace, P2::ProductSpace) =
    ProductSpace(tuple(P1.spaces..., P2.spaces...))
⊗(P::ProductSpace, V::Z2Space) = ProductSpace(tuple(P.spaces..., V))


# unit element with respect to the monoidal structure of taking tensor products
Base.one(V::VectorSpace) = one(typeof(V))
Base.one(::Type{<:ProductSpace}) = ProductSpace{0}(())
Base.one(::Type{Z2Space}) = ProductSpace{0}(())

Base.:^(V::Z2Space, N::Int) = ProductSpace{N}(ntuple(n -> V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n -> V, N)...)
function Base.literal_pow(::typeof(^), V::Z2Space, p::Val{N}) where {N}
    return ProductSpace{N}(ntuple(n -> V, p))
end

fuse(P::ProductSpace{0}) = oneunit(Z2Space)
fuse(P::ProductSpace) = fuse(P.spaces...)

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]

Base.iterate(P::ProductSpace, args...) = Base.iterate(P.spaces, args...)
Base.indexed_iterate(P::ProductSpace, args...) = Base.indexed_iterate(P.spaces, args...)

Base.eltype(::Type{<:ProductSpace}) = Z2Space
Base.eltype(P::ProductSpace) = eltype(typeof(P))

Base.IteratorEltype(::Type{<:ProductSpace}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ProductSpace}) = Base.HasLength()

Base.reverse(P::ProductSpace) = ProductSpace(reverse(P.spaces))



# Promotion and conversion
# ------------------------
Base.promote_rule(::Type{Z2Space}, ::Type{<:ProductSpace}) = ProductSpace


# ElementarySpace to ProductSpace
Base.convert(::Type{<:ProductSpace}, V::Z2Space) = ⊗(V)
