"""
    TreeTransformer

Supertype for structures containing the data for a tree transformation.
"""
abstract type TreeTransformer end

# struct TrivialTreeTransformer <: TreeTransformer end

struct AbelianTreeTransformer{T,N1,N2,N3,N4} <: TreeTransformer
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{T}
    structure_dst::FusionBlockStructure{N1,N2}
    structure_src::FusionBlockStructure{N3,N4}
end

function treetransformertype(Vdst, Vsrc)
    return AbelianTreeTransformer{Int,numout(Vdst),numin(Vdst),numout(Vsrc),numin(Vsrc)}
end

function TreeTransformer(transform::Function, Vsrc::HomSpace,
                         Vdst::HomSpace)
    structure_dst = fusionblockstructure(Vdst)
    structure_src = fusionblockstructure(Vsrc)
    return TreeTransformer(transform, structure_src, structure_dst)
end

function TreeTransformer(transform::Function, tsrc::TensorMap, tdst::TensorMap)
    return TreeTransformer(transform, tsrc.structure, tdst.structure)
end

function TreeTransformer(transform::Function,
                         structure_src::FusionBlockStructure,
                         structure_dst::FusionBlockStructure)
    rows = Int[]
    cols = Int[]
    vals = Int[]

    for (row, (f1, f2)) in enumerate(structure_src.fusiontreelist)
        for ((f3, f4), coeff) in transform(f1, f2)
            col = structure_dst.fusiontreeindices[(f3, f4)]
            push!(rows, row)
            push!(cols, col)
            push!(vals, coeff)
        end
    end

    # if FusionStyle(I) isa UniqueFusion
        return AbelianTreeTransformer(rows, cols, vals, structure_dst, structure_src)
    # else
    #     ldst = length(structure_dst.fusiontreelist)
    #     lsrc = length(structure_src.fusiontreelist)
    #     matrix = sparse(rows, cols, vals, ldst, lsrc)
    #     return GenericTreeTransformer(matrix, structure_dst, structure_src)
    # end
end

for (transform, transformer) in ((:permute, :permuter),)
    # ((:permute, :permuter), (:transpose, :transposer))
    treetransformer = Symbol("tree", transformer)

    @eval begin
        function $treetransformer(::AbstractTensorMap, ::AbstractTensorMap, p::Index2Tuple)
            return fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
        end
        function $treetransformer(tdst::TensorMap, tsrc::TensorMap, p::Index2Tuple)
            fusiontreetransform(f1, f2) = $transform(f1, f2, p...)
            return TreeTransformer(fusiontreetransform, tsrc, tdst)
        end
    end
end
