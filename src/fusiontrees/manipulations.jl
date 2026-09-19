fusiontreedict() = SingletonDict


@inline function split(f::FusionTree{N}, M::Int) where {N}
    if M > N || M < 0
        throw(ArgumentError("M should be between 0 and N = $N"))
    elseif M === N
        (f, FusionTree{1}((f.coupled,), f.coupled))
    elseif M === 1
        f₁ = FusionTree{1}((f.uncoupled[1],), f.uncoupled[1])
        f₂ = FusionTree{N}(f.uncoupled, f.coupled)
        return f₁, f₂
    elseif M === 0
        f₁ = FusionTree{0}((), one(Z2Irrep))
        uncoupled2 = (one(Z2Irrep), f.uncoupled...)
        coupled2 = f.coupled
        return f₁, FusionTree{N+1}(uncoupled2, coupled2)
    else
        uncoupled1 = ntuple(n -> f.uncoupled[n], M)
        coupled1 = couple(uncoupled1)

        uncoupled2 = ntuple(N - M + 1) do n
            return n == 1 ? coupled1 : f.uncoupled[M + n - 1]
        end
        coupled2 = f.coupled
        f₁ = FusionTree{M}(uncoupled1, coupled1)
        f₂ = FusionTree{N-M+1}(uncoupled2, coupled2)
        return f₁, f₂
    end
end

function permute(f1::FusionTree, f2::FusionTree,
                            p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {N₁, N₂}
    uncoupled = (f1.uncoupled..., dual.(f2.uncoupled)...)
    uncoupled1′, uncoupled2′ = TupleTools.getindices(uncoupled, p1), TupleTools.getindices(uncoupled, p2)
    uncoupled2′ = ntuple(i->dual(uncoupled2′[i]), Val(N₂))

    coupled1′ = couple(uncoupled1′)
    coupled2′ = couple(uncoupled2′)

    f1′ = FusionTree(uncoupled1′, coupled1′)
    f2′ = FusionTree(uncoupled2′, coupled2′)
    # compute the sign
    coeff = (coupled1′ == coupled2′) ? 1 : 0
    return fusiontreedict()((f1′, f2′) => coeff)
end

"""
    merge(f1::FusionTree{N₁}, f2::FusionTree{N₂}, c::Z2Irrep, μ::Int = 1)

Merge two fusion trees `f1` and `f2` into a single fusion tree with uncoupled legs
`(f1.uncoupled..., f2.uncoupled...)` and total charge `c`; returns a dictionary
`tree => coefficient`. For `Z2Irrep` the fusion is unique, so the result contains
a single tree with coefficient 1 (provided `couple((f1.coupled, f2.coupled)) == c`,
otherwise it is empty).
"""
function Base.merge(f1::FusionTree{N₁}, f2::FusionTree{N₂}, c::Z2Irrep,
                    μ::Int = 1) where {N₁, N₂}
    μ == 1 || return Dict{FusionTree{N₁ + N₂}, Int}()
    out = Dict{FusionTree{N₁ + N₂}, Int}()
    if c == couple((f1.coupled, f2.coupled))
        tree = FusionTree{N₁ + N₂}((f1.uncoupled..., f2.uncoupled...), c)
        out[tree] = 1
    end
    return out
end
