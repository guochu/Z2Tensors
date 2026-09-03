

module Z2TensorsCUDAExt

using Z2Tensors
const TK = Z2Tensors
using Z2Tensors: AdjointTensorMap, RealOrComplexFloat, NoTruncation,
        SectorDict, _empty_svdtensors, _compute_truncdim, _compute_truncerr


using CUDA, cuTENSOR
using LinearAlgebra: LinearAlgebra
using MatrixAlgebraKit:MatrixAlgebraKit, qr_compact, qr_compact!, lq_compact, lq_compact!, 
        svd_compact, svd_compact!, CUSOLVER_HouseholderQR, CUSOLVER_QRIteration, LQViaTransposedQR
MAK = MatrixAlgebraKit

export tocu, fromcu, CuTensorMap, CuDiagonalTensorMap, CuAdjointTensorMap




const CuTensorMap{T,N₁,N₂,A} = TensorMap{T,N₁,N₂,A} where {T,N₁,N₂,A<:CuArray{T,1}}
const CuAdjointTensorMap{T,N₁,N₂,A} = AdjointTensorMap{T,N₁,N₂,A} where {T,N₁,N₂,A<:CuTensorMap}

# convert a TensorMap to CuTensor
# TODO: togpu, fromgpu
function tocu(t::TensorMap{T,N₁,N₂,A}) where {T,N₁,N₂,A<:Array}
    data = CuArray(t.data)
    return TensorMap{T,N₁,N₂,typeof(data)}(data, t.space)
end
function fromcu(t::CuTensorMap{T,N₁,N₂}) where {T,N₁,N₂}
    data = Array(t.data)
    return TensorMap{T,N₁,N₂,typeof(data)}(data, t.space)
end


function TK.scalar(t::CuTensorMap)
    return dim(codomain(t)) == dim(domain(t)) == 1 ?
        Array(first(blocks(t))[2])[1, 1] : throw(DimensionMismatch())
end
function Base.copy!(tdst::CuTensorMap, tsrc::CuAdjointTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(bdst, bsrc)
    end
    return tdst
end


const CuDiagonalTensorMap{T,A} = DiagonalTensorMap{T,A} where {T,A<:CuArray{T,1}}

function tocu(t::DiagonalTensorMap{T,A}) where {T,A<:Array}
    data = CuArray(t.data)
    return DiagonalTensorMap{T,typeof(data)}(data, t.domain)
end
function fromcu(t::CuDiagonalTensorMap{T}) where {T}
    data = Array(t.data)
    return DiagonalTensorMap{T,typeof(data)}(data, t.domain)
end

   

include("mak.jl")
include("factorizations.jl")


end
