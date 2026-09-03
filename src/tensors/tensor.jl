# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#


struct TensorMap{T<:Number,N₁,N₂,A<:DenseVector{T},
                 FBS<:FusionBlockStructure{N₁,N₂}} <: AbstractTensorMap{T,N₁,N₂}
    data::A
    space::TensorMapSpace{N₁,N₂}
    structure::FBS

    # constructors from data and space (structure is computed from the space)
    function TensorMap{T,N₁,N₂,A}(data::A,
                                    space::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂,
                                                                         A<:DenseVector{T}}
        structure = fusionblockstructure(space)
        return new{T,N₁,N₂,A,typeof(structure)}(data, space, structure)
    end

    # constructor from data, space and a precomputed structure
    function TensorMap{T,N₁,N₂,A,FBS}(data::A,
                                      space::TensorMapSpace{N₁,N₂},
                                      structure::FBS) where {T,N₁,N₂,
                                                             A<:DenseVector{T},
                                                             FBS<:FusionBlockStructure{N₁,N₂}}
        return new{T,N₁,N₂,A,FBS}(data, space, structure)
    end
end

# uninitialized constructor (outer, computes the structure from the space)
function TensorMap{T,N₁,N₂,A}(::UndefInitializer,
                                space::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂,
                                                                     A<:DenseVector{T}}
    structure = fusionblockstructure(space)
    data = A(undef, structure.totaldim)
    if !isbitstype(T)
        zerovector!(data)
    end
    return TensorMap{T,N₁,N₂,A,typeof(structure)}(data, space, structure)
end

function TensorMap{T,N₁,N₂,A,FBS}(::UndefInitializer,
                                    space::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂,
                                                                         A<:DenseVector{T},
                                                                         FBS<:FusionBlockStructure{N₁,N₂}}
    structure = fusionblockstructure(space)
    data = A(undef, structure.totaldim)
    if !isbitstype(T)
        zerovector!(data)
    end
    return TensorMap{T,N₁,N₂,A,FBS}(data, space, structure)
end

const Tensor{T,N,A} = TensorMap{T,N,0,A}

function tensormaptype(N₁, N₂, TorA::Type)
    FBS = fusionblockstructuretype(N₁, N₂)
    if TorA <: Number
        return TensorMap{TorA,N₁,N₂,Vector{TorA},FBS}
    elseif TorA <: DenseVector
        return TensorMap{scalartype(TorA),N₁,N₂,TorA,FBS}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:DenseVector{<:Number}`"))
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::TensorMap) = t.space

storagetype(::Type{<:TensorMap{T,N₁,N₂,A}}) where {T,N₁,N₂,A<:DenseVector{T}} = A

dim(t::TensorMap) = length(t.data)

# General TensorMap constructors
#--------------------------------
# undef constructors
function TensorMap{T}(::UndefInitializer, V::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂}
    return TensorMap{T,N₁,N₂,Vector{T}}(undef, V)
end
function TensorMap{T}(::UndefInitializer, codomain::TensorSpace,
                      domain::TensorSpace) where {T}
    return TensorMap{T}(undef, codomain ← domain)
end
function Tensor{T}(::UndefInitializer, V::TensorSpace) where {T}
    return TensorMap{T}(undef, V ← one(V))
end

# constructor starting from vector = independent data (N₁ + N₂ = 1 is special cased below)
# documentation is captured by the case where `data` is a general array
# here, we force the `T` argument to distinguish it from the more general constructor below
function TensorMap(data::DenseVector{T}, space::TensorMapSpace{N₁,N₂},
                   structure::FusionBlockStructure{N₁,N₂}) where {T,N₁,N₂}
    return TensorMap{T,N₁,N₂,typeof(data),typeof(structure)}(data, space, structure)
end
function TensorMap{T}(data::A,
                      V::TensorMapSpace{N₁,N₂}) where {T,N₁,N₂,A<:DenseVector{T}}
    return TensorMap{T,N₁,N₂,A}(data, V)
end
function TensorMap{T}(data::DenseVector{T}, codomain::TensorSpace,
                      domain::TensorSpace) where {T}
    return TensorMap(data, codomain ← domain)
end

# constructor starting from block data
function TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix},
                   V::TensorMapSpace{N₁,N₂}) where {N₁,N₂}
    T = eltype(valtype(data))
    t = TensorMap{T}(undef, V)
    for (c, b) in blocks(t)
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
        datac = data[c]
        size(datac) == size(b) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
        copy!(b, datac)
    end
    for (c, b) in data
        c ∈ blocksectors(t) || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
    end
    return t
end
function TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix}, codom::TensorSpace,
                   dom::TensorSpace)
    return TensorMap(data, codom ← dom)
end


for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(codomain::TensorSpace,
                             domain::TensorSpace=one(codomain))
            return Base.$fname(codomain ← domain)
        end
        function Base.$fname(::Type{T}, codomain::TensorSpace,
                             domain::TensorSpace=one(codomain)) where {T}
            return Base.$fname(T, codomain ← domain)
        end
        Base.$fname(V::TensorMapSpace) = Base.$fname(Float64, V)
        function Base.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randf in (:rand, :randn, :randexp)
    randfun = GlobalRef(Random, randf)
    randfun! = GlobalRef(Random, Symbol(randf, :!))

    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(codomain::TensorSpace,
                          domain::TensorSpace)
            return $randfun(codomain ← domain)
        end
        function $randfun(::Type{T}, codomain::TensorSpace,
                          domain::TensorSpace) where {T}
            return $randfun(T, codomain ← domain)
        end
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          codomain::TensorSpace,
                          domain::TensorSpace) where {T}
            return $randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        $randfun(codomain::TensorSpace) = $randfun(codomain ← one(codomain))
        function $randfun(::Type{T}, codomain::TensorSpace) where {T}
            return $randfun(T, codomain ← one(codomain))
        end
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          codomain::TensorSpace) where {T}
            return $randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        $randfun(V::TensorMapSpace) = $randfun(Float64, V)
        function $randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return $randfun(rng, Float64, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, V::TensorMapSpace) where {T}
            return $randfun(Random.default_rng(), T, V)
        end
        $randfun!(t::AbstractTensorMap) = $randfun!(Random.default_rng(), t)

        # implementation
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end

        function $randfun!(rng::Random.AbstractRNG, t::AbstractTensorMap)
            for (_, b) in blocks(t)
                $randfun!(rng, b)
            end
            return t
        end
    end
end

function TensorMap(data::AbstractVector, V::TensorMapSpace{N₁,N₂}) where {N₁,N₂}
    T = eltype(data)
    @assert length(data) == dim(V)
    if data isa DenseVector # refer to specific data-capturing constructor
        return TensorMap{T}(data, V)
    else
        return TensorMap{T}(collect(data), V)
    end
end
function TensorMap(data::AbstractArray, codom::TensorSpace, dom::TensorSpace)
    return TensorMap(data, codom ← dom)
end
function Tensor(data::AbstractArray, codom::TensorSpace)
    return TensorMap(data, codom ← one(codom))
end




# Efficient copy constructors
#-----------------------------
Base.copy(t::TensorMap) = TensorMap(copy(t.data), t.space, t.structure)

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol,Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String,Any}()
    for (c, b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
block(t::TensorMap, c::Sector) = blocks(t)[c]

fusionblockstructure(t::TensorMap) = t.structure

blocks(t::TensorMap) = BlockIterator(t, t.structure.blockstructure)

function blocktype(::Type{TT}) where {TT<:TensorMap}
    A = storagetype(TT)
    T = eltype(A)
    return Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}
end

function Base.iterate(iter::BlockIterator{<:TensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, (sz, r)), newstate = next
    return c => reshape(view(iter.t.data, r), sz), newstate
end

function Base.getindex(iter::BlockIterator{<:TensorMap}, c::Sector)
    sectortype(iter.t) === typeof(c) || throw(SectorMismatch())
    (d₁, d₂), r = get(iter.structure, c) do
        # is s is not a key, at least one of the two dimensions will be zero:
        # it then does not matter where exactly we construct a view in `t.data`,
        # as it will have length zero anyway
        d₁′ = blockdim(codomain(iter.t), c)
        d₂′ = blockdim(domain(iter.t), c)
        l = d₁′ * d₂′
        return (d₁′, d₂′), 1:l
    end
    return reshape(view(iter.t.data, r), (d₁, d₂))
end

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
@inline function Base.getindex(t::TensorMap{T,N₁,N₂},
                               f₁::FusionTree{N₁},
                               f₂::FusionTree{N₂}) where {T,N₁,N₂}
    structure = fusionblockstructure(t)
    @boundscheck begin
        haskey(structure.fusiontreeindices, (f₁, f₂)) || throw(SectorMismatch())
    end
    @inbounds begin
        i = structure.fusiontreeindices[(f₁, f₂)]
        sz, str, offset = structure.fusiontreestructure[i]
        return StridedView(t.data, sz, str, offset)
    end
end


@propagate_inbounds function Base.setindex!(t::TensorMap{T,N₁,N₂},
                                            v,
                                            f₁::FusionTree{N₁},
                                            f₂::FusionTree{N₂}) where {T,N₁,N₂}
    return copy!(getindex(t, f₁, f₂), v)
end

@inline function Base.getindex(t::TensorMap, sectors::Tuple{I,Vararg{I}}) where {I<:Sector}
    I === sectortype(t) || throw(SectorMismatch("Not a valid sectortype for this tensor."))
    # FusionStyle(I) isa UniqueFusion ||
    #     throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    length(sectors) == numind(t) ||
        throw(ArgumentError("Number of sectors does not match."))
    s₁ = TupleTools.getindices(sectors, codomainind(t))
    s₂ = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = couple(s₁)
    @boundscheck begin
        c2 = couple(s₂)
        c2 == c1 || throw(SectorMismatch("Not a valid sector for this tensor"))
        hassector(codomain(t), s₁) && hassector(domain(t), s₂)
    end
    f₁ = FusionTree(s₁, c1)
    f₂ = FusionTree(s₂, c1)
    @inbounds begin
        return t[f₁, f₂]
    end
end
@propagate_inbounds function Base.getindex(t::TensorMap, sectors::Tuple)
    return t[map(sectortype(t), sectors)]
end

# Complex, real and imaginary parts
#-----------------------------------
for f in (:real, :imag, :complex)
    @eval begin
        function Base.$f(t::TensorMap)
            return TensorMap($f(t.data), space(t))
        end
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
function Base.convert(::Type{TensorMap}, t::AbstractTensorMap)
    return copy!(TensorMap{scalartype(t)}(undef, space(t)), t)
end

function Base.convert(TT::Type{<:TensorMap}, t::AbstractTensorMap)
    typeof(t) === TT && return t
    tnew = TT(undef, space(t))
    return copy!(tnew, t)
end

function Base.promote_rule(::Type{<:TT₁},
                           ::Type{<:TT₂}) where {N₁,N₂,
                                                 TT₁<:TensorMap{<:Any,N₁,N₂},
                                                 TT₂<:TensorMap{<:Any,N₁,N₂}}
    A = VectorInterface.promote_add(storagetype(TT₁), storagetype(TT₂))
    T = scalartype(A)
    return TensorMap{T,N₁,N₂,A,fusionblockstructuretype(N₁, N₂)}
end



function Base.empty(::Type{<:TensorMap{T,N₁,N₂,A}}) where {T,N₁,N₂,
                                                                A<:DenseVector{T}}
    space = fuse(ntuple(_ -> zero(Z2Space), N₁)) ← fuse(ntuple(_ -> zero(Z2Space), N₂))
    TensorMap{T,N₁,N₂,A}(undef, space)
end
Base.empty(t::TensorMap) = empty(typeof(t))

