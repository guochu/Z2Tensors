
struct HomSpace{N₁,N₂}
    codomain::ProductSpace{N₁}
    domain::ProductSpace{N₂}
end
codomain(W::HomSpace) = W.codomain
domain(W::HomSpace) = W.domain

dual(W::HomSpace) = HomSpace(dual(W.domain), dual(W.codomain))
function Base.adjoint(W::HomSpace)
    return HomSpace(W.domain, W.codomain)
end

Base.hash(W::HomSpace, h::UInt) = hash(domain(W), hash(codomain(W), h))
function Base.:(==)(W₁::HomSpace, W₂::HomSpace)
    return (W₁.codomain == W₂.codomain) && (W₁.domain == W₂.domain)
end

spacetype(W::HomSpace) = spacetype(typeof(W))
sectortype(W::HomSpace) = sectortype(typeof(W))

spacetype(::Type{<:HomSpace}) = Z2Space
sectortype(::Type{<:HomSpace}) = Z2Irrep

numout(W::HomSpace) = length(codomain(W))
numin(W::HomSpace) = length(domain(W))
numind(W::HomSpace) = numin(W) + numout(W)

const TensorSpace = Union{Z2Space,ProductSpace}
const TensorMapSpace{N₁,N₂} = HomSpace{N₁,N₂}

function Base.getindex(W::TensorMapSpace{N₁,N₂}, i) where {N₁,N₂}
    return i <= N₁ ? codomain(W)[i] : dual(domain(W)[i - N₁])
end

function ←(codom::ProductSpace, dom::ProductSpace)
    return HomSpace(codom, dom)
end
function ←(codom::Z2Space, dom::Z2Space)
    return HomSpace(ProductSpace(codom), ProductSpace(dom))
end
←(codom::VectorSpace, dom::VectorSpace) = ←(promote(codom, dom)...)
→(dom::VectorSpace, codom::VectorSpace) = ←(codom, dom)


function blocksectors(W::HomSpace)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)
    # TODO: is sort! still necessary now that blocksectors of ProductSpace is sorted?
    if N₂ <= N₁
        return sort!(filter!(c -> hasblock(codom, c), blocksectors(dom)))
    else
        return sort!(filter!(c -> hasblock(dom, c), blocksectors(codom)))
    end
end

hasblock(W::HomSpace, c::Sector) = hasblock(codomain(W), c) && hasblock(domain(W), c)

function dim(W::HomSpace)
    d = 0
    for c in blocksectors(W)
        d += blockdim(codomain(W), c) * blockdim(domain(W), c)
    end
    return d
end

# Operations on HomSpaces
# -----------------------

function permute(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    p = (p₁..., p₂...)
    TupleTools.isperm(p) && length(p) == numind(W) ||
        throw(ArgumentError("$((p₁, p₂)) is not a valid permutation for $(W)"))
    return select(W, (p₁, p₂))
end

function select(W::HomSpace, (p₁, p₂)::Index2Tuple{N₁,N₂}) where {N₁,N₂}
    cod = ProductSpace{N₁}(map(n -> W[n], p₁))
    dom = ProductSpace{N₂}(map(n -> dual(W[n]), p₂))
    return cod ← dom
end

function compose(W::HomSpace, V::HomSpace)
    domain(W) == codomain(V) || throw(SpaceMismatch("$(domain(W)) ≠ $(codomain(V))"))
    return HomSpace(codomain(W), domain(V))
end

function insertleftunit(W::HomSpace, i::Int = numind(W) + 1;
                        conj::Bool = false, dual::Bool = false)
    if i <= numout(W)
        return insertleftunit(codomain(W), i; conj, dual) ← domain(W)
    else
        return codomain(W) ← insertleftunit(domain(W), i - numout(W); conj, dual)
    end
end

function insertrightunit(W::HomSpace, i::Int = numind(W);
                         conj::Bool = false, dual::Bool = false)
    if i <= numout(W)
        return insertrightunit(codomain(W), i; conj, dual) ← domain(W)
    else
        return codomain(W) ← insertrightunit(domain(W), i - numout(W); conj, dual)
    end
end

function removeunit(W::HomSpace, i::Int)
    if i <= numout(W)
        return removeunit(codomain(W), i) ← domain(W)
    else
        return codomain(W) ← removeunit(domain(W), i - numout(W))
    end
end

# Block and fusion tree ranges: structure information for building tensors
#--------------------------------------------------------------------------
struct FusionBlockStructure{N₁,N₂,N}
    totaldim::Int
    blockstructure::SectorDict{Tuple{Tuple{Int,Int},UnitRange{Int}}}
    fusiontreelist::Vector{Tuple{FusionTree{N₁},FusionTree{N₂}}}
    fusiontreestructure::Vector{Tuple{NTuple{N,Int},NTuple{N,Int},Int}}
    fusiontreeindices::FusionTreeDict{Tuple{FusionTree{N₁},FusionTree{N₂}},Int}

    function FusionBlockStructure{N₁,N₂}(totaldim, blockstructure, fusiontreelist,
                                         fusiontreestructure,
                                         fusiontreeindices) where {N₁,N₂}
        return new{N₁,N₂,N₁ + N₂}(totaldim, blockstructure, fusiontreelist,
                                  fusiontreestructure, fusiontreeindices)
    end
end

"""
    fusionblockstructuretype(N₁, N₂)

Compute the concrete type of the [`FusionBlockStructure`](@ref) associated with a
`TensorMapSpace` with `N₁` output and `N₂` input slots.
"""
fusionblockstructuretype(N₁::Int, N₂::Int) = FusionBlockStructure{N₁,N₂,N₁ + N₂}

function fusionblockstructure(W::HomSpace)
    codom = codomain(W)
    dom = domain(W)
    N₁ = length(codom)
    N₂ = length(dom)

    # output structure
    blockstructure = SectorDict{Tuple{Tuple{Int,Int},UnitRange{Int}}}() # size, range
    fusiontreelist = Vector{Tuple{FusionTree{N₁},FusionTree{N₂}}}()
    fusiontreestructure = Vector{Tuple{NTuple{N₁ + N₂,Int},NTuple{N₁ + N₂,Int},Int}}() # size, strides, offset

    # temporary data structures
    splittingtrees = Vector{FusionTree{N₁}}()
    splittingstructure = Vector{Tuple{Int,Int}}()

    # main computational routine
    blockoffset = 0
    for c in blocksectors(W)
        empty!(splittingtrees)
        empty!(splittingstructure)

        offset₁ = 0
        for f₁ in fusiontrees(codom, c)
            push!(splittingtrees, f₁)
            d₁ = dim(codom, f₁.uncoupled)
            push!(splittingstructure, (offset₁, d₁))
            offset₁ += d₁
        end
        blockdim₁ = offset₁
        strides = (1, blockdim₁)

        offset₂ = 0
        for f₂ in fusiontrees(dom, c)
            s₂ = f₂.uncoupled
            d₂ = dim(dom, s₂)
            for (f₁, (offset₁, d₁)) in zip(splittingtrees, splittingstructure)
                push!(fusiontreelist, (f₁, f₂))
                totaloffset = blockoffset + offset₂ * blockdim₁ + offset₁
                subsz = (dims(codom, f₁.uncoupled)..., dims(dom, f₂.uncoupled)...)
                @assert !any(isequal(0), subsz)
                substr = _subblock_strides(subsz, (d₁, d₂), strides)
                push!(fusiontreestructure, (subsz, substr, totaloffset))
            end
            offset₂ += d₂
        end
        blockdim₂ = offset₂
        blocksize = (blockdim₁, blockdim₂)
        blocklength = blockdim₁ * blockdim₂
        blockrange = (blockoffset + 1):(blockoffset + blocklength)
        blockoffset = last(blockrange)
        blockstructure[c] = (blocksize, blockrange)
    end

    fusiontreeindices = sizehint!(FusionTreeDict{Tuple{FusionTree{N₁},FusionTree{N₂}},Int}(),
                                  length(fusiontreelist))
    for (i, f₁₂) in enumerate(fusiontreelist)
        fusiontreeindices[f₁₂] = i
    end
    totaldim = blockoffset
    structure = FusionBlockStructure{N₁,N₂}(totaldim, blockstructure,
                                            fusiontreelist, fusiontreestructure,
                                            fusiontreeindices)
    return structure
end

function _subblock_strides(subsz, sz, str)
    sz_simplify = Strided.StridedViews._simplifydims(sz, str)
    return Strided.StridedViews._computereshapestrides(subsz, sz_simplify...)
end

# Diagonal ranges
#----------------
# TODO: is this something we want to cache?
function diagonalblockstructure(W::HomSpace)
    ((numin(W) == numout(W) == 1) && domain(W) == codomain(W)) ||
        throw(SpaceMismatch("Diagonal only support on V←V with a single space V"))
    structure = SectorDict{UnitRange{Int}}() # range
    offset = 0
    dom = domain(W)[1]
    for c in blocksectors(W)
        d = dim(dom, c)
        structure[c] = offset .+ (1:d)
        offset += d
    end
    return structure
end
