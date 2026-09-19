push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/Z2Tensors/src")


using Test
using TestExtras
using Random
using Combinatorics
using TensorOperations
using Base.Iterators: take, product
using LinearAlgebra: LinearAlgebra

using Z2Tensors
using Z2Tensors: fusiontensor#, pentagon_equation, hexagon_equation # ProductSector, 
const TK = Z2Tensors

Random.seed!(1234)

smallset(::Type{Z2Irrep}) = (Z2Irrep(0), Z2Irrep(1))
function randsector(::Type{Z2Irrep})
    s = collect(smallset(Z2Irrep))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(::Type{Z2Irrep})
    try
        fusiontensor(one(Z2Irrep), one(Z2Irrep), one(Z2Irrep))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

sectorlist = (Z2Irrep,)

# spaces
VZ2 = (Z2Space(0 => 1, 1 => 1),
        Z2Space(0 => 1, 1 => 2)',
        Z2Space(0 => 3, 1 => 2)',
        Z2Space(0 => 2, 1 => 3),
        Z2Space(0 => 2, 1 => 5))

Ti = time()



include("convert.jl")

include("spaces.jl")
include("tensors.jl")
include("diagonal.jl")

# plain-array factorization utilities (ported from TEMPO)
include("tensorfactorizations.jl")

# plain-array tensor operations (permute, tie, isometry, kron)
include("tensoroperations.jl")

# consistency with TensorKit.jl
include("compare_tensorkit.jl")

