# Tests for the plain-array factorization utilities ported from TEMPO
# (src/tensorops/tensorfactorizations.jl), cf. TEMPO/test/api/truncation.jl
# and TEMPO/test/api/linalg.jl

@testset "truncation schemes (matrix)" begin
    @test truncdim(3) isa TK.TruncationDimension
    @test truncdim(D = 4) isa TK.TruncationDimension
    @test truncdim(D = 4).dim == 4
    @test truncrelerr(ϵ = 1.0e-3) isa TK.TruncateRelError
    @test truncrelerr(ϵ = 1.0e-3).ϵ == 1.0e-3
    @test truncrelerr(1.0e-8) isa TK.TruncateRelError
    @test truncrelerr(1.0e-8).ϵ == 1.0e-8
    @test truncdimcutoff(D = 5, ϵ = 1.0e-3) isa TK.TruncateDimCutoff
    @test truncdimcutoff(5, 1.0e-3) isa TK.TruncateDimCutoff
    @test TK.NoTruncation() isa TK.TruncationScheme
    # add_back larger than D is rejected
    @test_throws ArgumentError truncdimcutoff(D = 2, ϵ = 1.0e-16, add_back = 5)
end

@testset "tsvd (matrix)              " begin
    a = randn(6, 5)
    # tsvd! destroys its input, use the non-mutating tsvd since a is reused below
    u, s, v, err = tsvd(a; trunc = truncdim(3))
    @test length(s) == 3
    @test err ≈ LinearAlgebra.norm(LinearAlgebra.svdvals(a)[4:end])
    u2, s2, v2, err2 = tsvd(a; trunc = truncrelerr(ϵ = 1.0e-10))
    @test length(s2) == 5
    @test err2 < 1.0e-8
    u3, s3, v3, err3 = tsvd(a; trunc = truncdimcutoff(D = 2, ϵ = 1.0e-10))
    @test length(s3) == 2
    @test err3 > 0
    u4, s4, v4, err4 = tsvd(a)
    @test length(s4) == 5
    @test err4 == 0.0
    @test u4 * LinearAlgebra.Diagonal(s4) * v4 ≈ a
    # tsvd! on a copy gives the same result
    u5, s5, v5, err5 = tsvd!(copy(a); trunc = truncdim(3))
    @test s5 ≈ s
end

# The ϵ of `truncrelerr` and `truncdimcutoff` is measured on the *normalized*
# vector of singular values, i.e. σᵢ (sorted decreasingly) is kept iff σᵢ > ϵ·‖σ‖₂.
@testset "truncation semantics       " begin
    a = randn(8, 6)
    s = LinearAlgebra.svdvals(a)   # decreasing
    n = norm(s)                    # ‖σ‖₂

    # relerr keeps exactly the singular values with σᵢ > ϵ·‖σ‖₂ ...
    ϵ = 0.2
    d = count(>(ϵ * n), s)
    u1, s1, v1, err1 = tsvd(a; trunc = truncrelerr(ϵ))
    @test length(s1) == d
    @test all(>(ϵ * n), s1)
    # ... which is the same as keeping the largest `d` singular values
    u2, s2, v2, err2 = tsvd(a; trunc = truncdim(d))
    @test s1 == s2 && err1 ≈ err2
    # on a pre-normalized matrix, relerr(ϵ) is an absolute cutoff at ϵ
    an = a ./ n
    sn = LinearAlgebra.svdvals(an)
    _, s3, _, _ = tsvd(an; trunc = truncrelerr(ϵ))
    @test s3 ≈ sn[sn .> ϵ]
    # strict, stable boundary: ϵ → nextfloat(ϵ) drops exactly the marginal values
    _, s4, _, _ = tsvd(a; trunc = truncrelerr(nextfloat(ϵ)))
    @test length(s4) == count(>(nextfloat(ϵ) * n), s)

    # dimcutoff with a non-binding D is exactly relerr; its error is *relative*
    u5, s5, v5, err5 = tsvd(a; trunc = truncdimcutoff(D = 20, ϵ = ϵ))
    _, s6, _, err6 = tsvd(a; trunc = truncrelerr(ϵ))
    @test s5 == s6
    @test err5 ≈ norm(s[(d + 1):end]) / n  # relative truncation error
    @test err6 ≈ norm(s[(d + 1):end])      # relerr reports the absolute tail norm

    # add_back keeps at least that many singular values, but never more than D
    _, s7, _, _ = tsvd(a; trunc = truncdimcutoff(D = 10, ϵ = 0.9, add_back = 3))
    @test length(s7) == max(3, count(>(0.9 * n), s))
    _, s8, _, _ = tsvd(a; trunc = truncdimcutoff(D = 2, ϵ = 1.0e-16, add_back = 2))
    @test length(s8) == 2

    # truncdim error: the 2-norm of the discarded tail; NoTruncation: nothing dropped
    _, s9, _, err9 = tsvd(a; trunc = truncdim(4))
    @test err9 ≈ norm(s[5:end])
    _, st, _, errt = tsvd(a)
    @test length(st) == 6 && errt == 0.0
    # truncerr: keep the largest set whose tail norm stays below ϵ
    _, serr, _, ererr = tsvd(a; trunc = truncerr(0.5 * norm(s)))
    @test ererr <= 0.5 * norm(s) &&
          ererr + norm(serr[end]) > 0.5 * norm(s)

    # keyword constructors agree with the convenience functions
    @test TK.TruncateRelError(ϵ = 0.1) == truncrelerr(0.1)
    @test TK.TruncateDimCutoff(D = 5, ϵ = 0.1) == truncdimcutoff(D = 5, ϵ = 0.1)

    # tsvd! (in place) agrees with tsvd
    ub, sb, vb, errb = tsvd!(copy(a); trunc = truncdimcutoff(D = 4, ϵ = 1.0e-3))
    _, snb, _, errnb = tsvd(a; trunc = truncdimcutoff(D = 4, ϵ = 1.0e-3))
    @test sb == snb && errb ≈ errnb
end

@testset "renyi_entropy              " begin
    v = [0.25, 0.75]
    @test renyi_entropy(v) ≈ -(0.25 * log(0.25) + 0.75 * log(0.75))
    @test renyi_entropy(v; α = 2) ≈ -log(0.25^2 + 0.75^2)
    @test renyi_entropy([1.0]) == 0.0
    @test_throws ArgumentError renyi_entropy([0.5, 0.6])  # not normalized
    @test_throws ArgumentError renyi_entropy([-0.5, 1.5]) # negative entries
    # on normalized squared singular values
    u, s, v2, _ = tsvd!(randn(5, 5))
    p = s .^ 2 ./ sum(s .^ 2)
    @test renyi_entropy(p) > 0
end

@testset "isometry                   " begin
    @test isometry(3) == LinearAlgebra.Matrix(LinearAlgebra.I, 3, 3)
    @test isometry(3) isa Matrix{Float64}
    @test isometry(ComplexF64, 3) == LinearAlgebra.Matrix(LinearAlgebra.I, 3, 3)
    @test isometry(ComplexF64, 3) isa Matrix{ComplexF64}
    i34 = isometry(3, 4)
    @test i34 == [1 0 0 0; 0 1 0 0; 0 0 1 0]
    @test i34 * i34' == LinearAlgebra.Matrix(LinearAlgebra.I, 3, 3)
    i32 = isometry(ComplexF64, 3, 2)
    @test i32 == [1 0; 0 1; 0 0]
    @test i32' * i32 == LinearAlgebra.Matrix(LinearAlgebra.I, 2, 2)
end

@testset "permute (plain array)      " begin
    a = randn(3, 4)
    @test permute(a, (2, 1)) ≈ permutedims(a, (2, 1))
    @test permute(a, (1,), (2,)) ≈ a
    t = randn(2, 3, 4)
    @test permute(t, (1, 2), (3,)) ≈ permute(t, (1, 2, 3))
    @test TK.tie(t, (2, 1)) ≈ reshape(t, 6, 4)
end

@testset "tsvd and tsvd! (tensor)    " begin
    a = randn(ComplexF64, 6, 5)
    ac = copy(a)
    # in-place plain-tensor version (also covers permute + reconstruction)
    t = randn(2, 3, 4)
    tc = copy(t)
    u, s, v, err = tsvd!(t, (1, 2), (3,))
    md = length(s)
    @test size(u) == (2, 3, md)
    @test size(v) == (md, 4)
    r = zeros(size(tc))
    for k in 1:md
        r += u[:, :, k] .* s[k] .* reshape(v[k, :], 1, 1, 4)
    end
    @test r ≈ tc
    @test err == 0.0
    # non-mutating tensor version
    u2, s2, v2, err2 = tsvd(t, (1,), (2, 3))
    @test t == tc
    @test length(s2) == 2
    @test err2 == 0.0

    for alg in (SVD(), SDD())
        ua, sa, va, erra = tsvd(a; alg = alg)
        @test a == ac                       # input not modified
        @test erra == 0.0
        @test ua * LinearAlgebra.Diagonal(sa) * va ≈ a
        @test all(sa .>= 0)
        # alg keyword also available for tsvd!
        u2a, s2a, v2a, err2a = tsvd!(copy(a); alg = alg)
        @test s2a ≈ sa
    end
    # SVD/SDD drivers give the same singular values
    _, s_svd, _, _ = tsvd(a; alg = SVD())
    _, s_sdd, _, _ = tsvd(a; alg = SDD())
    @test s_svd ≈ s_sdd
end

@testset "leftorth and rightorth     " begin
    A = randn(ComplexF64, 6, 4)
    Ac = copy(A)
    # default algs
    Q, R = leftorth(A)
    @test A == Ac
    @test Q * R ≈ A
    @test Q' * Q ≈ LinearAlgebra.Matrix(LinearAlgebra.I, 4, 4)
    L, Q2 = rightorth(A)
    @test A == Ac
    @test L * Q2 ≈ A
    @test Q2 * Q2' ≈ LinearAlgebra.Matrix(LinearAlgebra.I, 4, 4)
    # all algorithms
    for alg in (QR(), QRpos(), SVD(), SDD(), Polar())
        Q, R = leftorth(A; alg = alg)
        @test A == Ac
        @test Q * R ≈ A
        @test Q' * Q ≈ LinearAlgebra.Matrix(LinearAlgebra.I, 4, 4)
    end
    for alg in (LQ(), LQpos(), SVD(), SDD())
        L, Q = rightorth(A; alg = alg)
        @test A == Ac
        @test L * Q ≈ A
        @test Q * Q' ≈ LinearAlgebra.Matrix(LinearAlgebra.I, 4, 4)
    end
    # Polar right-orthogonalization requires a wide matrix
    Aw = randn(ComplexF64, 4, 6)
    L, Q = rightorth(Aw; alg = Polar())
    @test L * Q ≈ Aw
    @test Q * Q' ≈ LinearAlgebra.Matrix(LinearAlgebra.I, 4, 4)
    # SVD-based orthogonalization with atol truncates small singular values
    B = LinearAlgebra.Diagonal([1.0, 1.0e-12, 1.0])
    Qb, Rb = leftorth(Matrix(B); alg = SVD(), atol = 1.0e-9)
    @test size(Qb, 2) == 2
    @test Qb * Rb ≈ B
    # plain-tensor versions
    T = randn(4, 3, 2)
    Tc = copy(T)
    u, v = leftorth(T, (1, 2), (3,))
    @test T == Tc
    s = size(v, 1)
    @test size(u) == (4, 3, s)
    @test size(v) == (s, 2)
    @test reshape(u, :, s)' * reshape(u, :, s) ≈
          LinearAlgebra.Matrix(LinearAlgebra.I, s, s)
    @test reshape(reshape(u, :, s) * reshape(v, s, :), 4, 3, 2) ≈ Tc
    T = randn(4, 3, 2)
    Tc = copy(T)
    u, v = rightorth(T, (1,), (2, 3))
    @test T == Tc
    s = size(v, 1)
    @test size(u) == (4, s)
    @test size(v) == (s, 3, 2)
    @test reshape(v, s, :) * reshape(v, s, :)' ≈
          LinearAlgebra.Matrix(LinearAlgebra.I, s, s)
    @test reshape(reshape(u, :, s) * reshape(v, s, :), 4, 3, 2) ≈ Tc
end
