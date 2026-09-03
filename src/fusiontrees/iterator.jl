
function couple(uncoupled::NTuple{N,Z2Irrep}) where {N}
    ⊗(uncoupled...)
end
function couple(uncoupled::NTuple{0})
    one(Z2Irrep)
end



function fusiontrees(uncoupled::NTuple{N,Z2Irrep}, coupled::Z2Irrep) where {N}
    if couple(uncoupled) == coupled
        return (FusionTree{N}(uncoupled, coupled), )
    else
        return ()
    end
end
function fusiontrees(uncoupleds::Tuple, coupled::Z2Irrep)
    trees = ((fusiontrees(uncoupled, coupled) for uncoupled in Iterators.product(uncoupleds...))...,)
    return TupleTools.flatten(trees)
end
