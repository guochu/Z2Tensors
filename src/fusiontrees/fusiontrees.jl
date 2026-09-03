
struct FusionTree{N}
    uncoupled::NTuple{N,Z2Irrep}
    coupled::Z2Irrep
    function FusionTree{N}(uncoupled::NTuple{N,Z2Irrep}, coupled::Z2Irrep) where {N}
        return new{N}(uncoupled, coupled)
    end
end

function FusionTree(uncoupled::NTuple{N,Z2Irrep}, coupled::Z2Irrep = unit(Z2Irrep)) where {N}
    return FusionTree{N}(uncoupled, coupled)
end



# Properties
sectortype(::Type{<:FusionTree}) = Z2Irrep
sectortype(f::FusionTree) = Z2Irrep
Base.length(f::FusionTree) = length(typeof(f))
Base.length(::Type{<:FusionTree{N}}) where {N} = N

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree, h::UInt)
    h = hash(f.coupled, hash(f.uncoupled, h))
    return h
end
function Base.:(==)(f₁::FusionTree{N}, f₂::FusionTree{N}) where {N}
    f₁.coupled == f₂.coupled || return false
    @inbounds for i in 1:N
        f₁.uncoupled[i] == f₂.uncoupled[i] || return false
    end
    return true
end
Base.:(==)(f₁::FusionTree, f₂::FusionTree) = false

# Facilitate getting correct fusion tree types
fusiontreetype(N::Int) = FusionTree{N}



# Manipulate fusion trees
include("manipulations.jl")

# Fusion tree iterators
include("iterator.jl")

# auxiliary routines
# _abelianinner: generate the inner indices for given outer indices in the abelian case
# _abelianinner(outer::Tuple{}) = ()
# function _abelianinner(outer::Tuple{I}) where {I<:Sector}
#     return isone(outer[1]) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I}) where {I<:Sector}
#     return outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I,I}) where {I<:Sector}
#     return isone(first(⊗(outer...))) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I,I,I,Vararg{I}}) where {I<:Sector}
#     c = first(outer[1] ⊗ outer[2])
#     return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
# end
