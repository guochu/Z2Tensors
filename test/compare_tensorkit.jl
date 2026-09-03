# Compare Z2Tensors behavior against TensorKit.jl
# ---------------------------------------------------------
# The same operations are performed on identical input data (block-wise
# filled with the same random values) and results are compared block-wise,
# ensuring that Z2Tensors is strictly consistent with TensorKit.

using Test
using Random
using TensorOperations: @tensor
import TensorKit as TKt
using TensorKit: Vect
import Z2Tensors as ZT
using Z2Tensors: Z2Space, Z2Irrep, charge
import Z2Tensors: ← as zarrow, ⊗ as zotimes
import TensorKit: ← as tarrow, ⊗ as totimes
using LinearAlgebra: LinearAlgebra, norm

# identical spaces in both packages
Z2W1 = Z2Space(0 => 2, 1 => 3)
Z2W2 = Z2Space(0 => 1, 1 => 2)
Z2W3 = Z2Space(0 => 1, 1 => 1)
TKW1 = Vect[TKt.Z2Irrep](0 => 2, 1 => 3)
TKW2 = Vect[TKt.Z2Irrep](0 => 1, 1 => 2)
TKW3 = Vect[TKt.Z2Irrep](0 => 1, 1 => 1)

# fill the blocks of a tensor with random numbers from a seeded RNG, such that
# tensors in both packages with the same spaces receive identical data
function randfill!(t, rng::AbstractRNG)
    getblocks(t) = t isa TKt.AbstractTensorMap ? TKt.blocks(t) : ZT.blocks(t)
    for (c, b) in getblocks(t)
        Random.randn!(rng, b)
    end
    return t
end

# integer charge of a sector from either package
sectorcharge(c) = c isa ZT.Sector ? ZT.charge(c) : Int(TKt.charge(c))

# construct a pair of tensors with identical data: fill the Z2Tensors tensor
# with random numbers, then copy its blocks into the TensorKit tensor
function paired_randn(z2space, tkspace, rng)
    z2t = randfill!(ZT.randn(ComplexF64, z2space), rng)
    tkt = TKt.randn(ComplexF64, tkspace)
    for ((c, b), (ct, bt)) in zip(ZT.blocks(z2t), TKt.blocks(tkt))
        sectorcharge(c) == sectorcharge(ct) ||
            error("sector mismatch: $c vs $ct")
        copy!(bt, b)
    end
    return z2t, tkt
end

getblocks(t) =
    t isa TKt.AbstractTensorMap ? [(c, b) for (c, b) in TKt.blocks(t)] :
    [(c, b) for (c, b) in ZT.blocks(t)]

# compare two tensors (one from each package) block-wise;
# sectors are compared via their integer charge
function checkblocks(z2t, tkt)
    bz = getblocks(z2t)
    bt = getblocks(tkt)
    length(bz) == length(bt) || return false
    for ((cz, b1), (ct, b2)) in zip(bz, bt)
        sectorcharge(cz) == sectorcharge(ct) || return false
        b1 ≈ b2 || return false
    end
    return true
end

@testset "Z2Tensors vs TensorKit consistency" begin
    rng = Random.Xoshiro(1234)

    #-------------------------------------------------------------------
    # basic properties (A : W2 ⊗ W3 ← W1, B : W1 ← W2, C : W1 ← W3)
    #-------------------------------------------------------------------
    A, TKA = paired_randn(zarrow(zotimes(Z2W2, Z2W3), Z2W1),
                          tarrow(totimes(TKW2, TKW3), TKW1), rng)
    B, TKB = paired_randn(zarrow(Z2W1, Z2W2), tarrow(TKW1, TKW2), rng)
    C, TKC = paired_randn(zarrow(Z2W1, Z2W3), tarrow(TKW1, TKW3), rng)
    @test checkblocks(A, TKA)
    @test ZT.dim(A) == TKt.dim(TKA)
    @test ZT.norm(A) ≈ TKt.norm(TKA)
    @test ZT.tr(A' * A) ≈ TKt.tr(TKA' * TKA)
    @test ZT.dot(B, B) ≈ TKt.dot(TKB, TKB)

    #-------------------------------------------------------------------
    # permute
    #-------------------------------------------------------------------
    Ap = ZT.permute(A, ((3, 1), (2,)))
    TKAp = TKt.permute(TKA, ((3, 1), (2,)))
    @test checkblocks(Ap, TKAp)

    #-------------------------------------------------------------------
    # @tensor: plain two-tensor contraction ( = )
    #-------------------------------------------------------------------
    D = @tensor d[3 4; 1] := A[2 3; 1] * B[4; 2]
    TKD = @tensor tkd[3 4; 1] := TKA[2 3; 1] * TKB[4; 2]
    @test checkblocks(D, TKD)

    #-------------------------------------------------------------------
    # @tensor: += accumulation
    #-------------------------------------------------------------------
    D0, TKD0 = paired_randn(zarrow(zotimes(Z2W3, Z2W1), Z2W1),
                            tarrow(totimes(TKW3, TKW1), TKW1), rng)
    D2 = copy(D0)
    TKD2 = copy(TKD0)
    @tensor D2[3 4; 1] += A[2 3; 1] * B[4; 2]
    @tensor TKD2[3 4; 1] += TKA[2 3; 1] * TKB[4; 2]
    @test checkblocks(D2, TKD2)

    #-------------------------------------------------------------------
    # @tensor: scalar coefficients (first, middle, last) with +=
    #-------------------------------------------------------------------
    for (α, mode) in ((2.5, :first), (2.5 + 1.5im, :middle), (-0.5 - 2.0im, :last))
        F = copy(D0)
        TKF = copy(TKD0)
        if mode == :first
            @tensor F[3 4; 1] += α * A[2 3; 1] * B[4; 2]
            @tensor TKF[3 4; 1] += α * TKA[2 3; 1] * TKB[4; 2]
        elseif mode == :middle
            @tensor F[3 4; 1] += A[2 3; 1] * α * B[4; 2]
            @tensor TKF[3 4; 1] += TKA[2 3; 1] * α * TKB[4; 2]
        else
            @tensor F[3 4; 1] += A[2 3; 1] * B[4; 2] * α
            @tensor TKF[3 4; 1] += TKA[2 3; 1] * TKB[4; 2] * α
        end
        @test checkblocks(F, TKF)
    end

    #-------------------------------------------------------------------
    # @tensor: multi-tensor contraction and parenthesis groupings
    #-------------------------------------------------------------------
    E, TKE = paired_randn(zarrow(Z2W3, Z2W1), tarrow(TKW3, TKW1), rng)
    Ed = @tensor ed[4 5; 1] := A[2 3; 1] * B[4; 2] * C[5; 3]
    TKEd = @tensor tked[4 5; 1] := TKA[2 3; 1] * TKB[4; 2] * TKC[5; 3]
    @test checkblocks(Ed, TKEd)

    Ep = @tensor ep[4 5; 1] := A[2 3; 1] * (B[4; 2] * C[5; 3])
    TKEp = @tensor tkep[4 5; 1] := TKA[2 3; 1] * (TKB[4; 2] * TKC[5; 3])
    @test checkblocks(Ep, TKEp)

    X = @tensor x[1 6; 4] := B[1; 2] * (A[2 3; 4] * (C[5; 3] * E[6; 5]))
    TKX = @tensor tkx[1 6; 4] := TKB[1; 2] * (TKA[2 3; 4] * (TKC[5; 3] * TKE[6; 5]))
    @test checkblocks(X, TKX)

    Y = @tensor y[1 6; 4] := (((B[1; 2] * A[2 3; 4]) * C[5; 3]) * E[6; 5])
    TKY = @tensor tky[1 6; 4] := (((TKB[1; 2] * TKA[2 3; 4]) * TKC[5; 3]) * TKE[6; 5])
    @test checkblocks(Y, TKY)

    #-------------------------------------------------------------------
    # @tensor: conjugated tensors (contraction over codomain / domain)
    #-------------------------------------------------------------------
    M, TKM = paired_randn(zarrow(Z2W1, Z2W1), tarrow(TKW1, TKW1), rng)
    C1 = @tensor c1[1; 2] := M[3; 1] * conj(M[3; 2])
    TKC1 = @tensor tkc1[1; 2] := TKM[3; 1] * conj(TKM[3; 2])
    @test checkblocks(C1, TKC1)

    C2 = @tensor c2[1; 2] := M[1; 3] * conj(M[2; 3])
    TKC2 = @tensor tkc2[1; 2] := TKM[1; 3] * conj(TKM[2; 3])
    @test checkblocks(C2, TKC2)

    # adjoint operand contracted with a plain tensor
    B2 = A'
    TKB2 = TKA'
    C3 = @tensor c3[1; 4] := B2[1; 2 3] * A[2 3; 4]
    TKC3 = @tensor tkc3[1; 4] := TKB2[1; 2 3] * TKA[2 3; 4]
    @test checkblocks(C3, TKC3)

    #-------------------------------------------------------------------
    # @tensor: adjoint tensors and scalar output
    #-------------------------------------------------------------------
    Ssc = @tensor s[] := A'[1; 2 3] * A[2 3; 1]
    TKSsc = @tensor tks[] := TKA'[1; 2 3] * TKA[2 3; 1]
    @test ZT.scalar(Ssc) ≈ TKt.scalar(TKSsc)

    #-------------------------------------------------------------------
    # factorizations: leftorth / tsvd (gauge-invariant comparisons)
    #-------------------------------------------------------------------
    Q, R = ZT.leftorth(A)
    TKQ, TKR = TKt.left_orth(TKA)
    @test checkblocks(A, Q * R)
    @test checkblocks(TKA, TKQ * TKR)
    @test norm(Q' * Q - one(Q' * Q)) < 1e-12
    @test norm(TKQ' * TKQ - one(TKQ' * TKQ)) < 1e-12

    U, S2, Vp, err = ZT.tsvd(A)
    TKU, TKS2, TKVp = TKt.svd_compact(TKA)
    @test checkblocks(A, U * S2 * Vp)
    @test checkblocks(TKA, TKU * TKS2 * TKVp)
    @test norm(U' * U - one(U' * U)) < 1e-12
    @test norm(TKU' * TKU - one(TKU' * TKU)) < 1e-12
    @test checkblocks(S2, TKS2) # singular values are unique up to ordering
end
