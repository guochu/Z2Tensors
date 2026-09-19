# Tests for the plain-array tensor operations in auxiliary/tensoroperations.jl
# (cf. TEMPO/test/api/linalg.jl) and the Kronecker product for Z2 tensors.

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
    @test tie(t, (2, 1)) ≈ reshape(t, 6, 4)
end

@testset "kron (plain array)         " begin
    a = randn(2, 3, 4)
    b = randn(3, 2, 5)
    c = kron(a, b)
    @test size(c) == (6, 6, 20)
    for i in (1, 2), ib in (1, 3), j in (1, 3), jb in (1, 2), k in (1, 4), kb in (1, 5)
        @test c[(i - 1) * 3 + ib, (j - 1) * 2 + jb, (k - 1) * 5 + kb] ≈ a[i, j, k] * b[ib, jb, kb]
    end
    # vectors and matrices keep the standard Kronecker semantics
    v1 = randn(4); v2 = randn(3)
    kv = kron(v1, v2)
    @test size(kv) == (length(v1) * length(v2),)
    @test all(kv[(i - 1) * length(v2) + j] ≈ v1[i] * v2[j]
              for i in eachindex(v1), j in eachindex(v2))
    M1 = randn(3, 4); M2 = randn(2, 5)
    K = kron(M1, M2)
    @test K isa Matrix{Float64}
    @test size(K) == (6, 20)
    @test K[1:2, 1:5] ≈ M1[1, 1] .* M2
    @test K[(2 - 1) * 2 .+ (1:2), (3 - 1) * 5 .+ (1:5)] ≈ M1[2, 3] .* M2
    @test eltype(kron(Matrix{ComplexF32}(M1), Matrix{Float32}(M2))) == ComplexF32
    @test eltype(kron(Matrix{ComplexF32}(M1), M2)) == ComplexF64
end

@testset "kron (Z2Tensor)            " begin
    W1 = Z2Space(0 => 1, 1 => 1)
    W2 = Z2Space(0 => 1, 1 => 2)
    W3 = Z2Space(0 => 3, 1 => 2)
    W4 = Z2Space(0 => 2, 1 => 3)
    W5 = Z2Space(0 => 2, 1 => 5)
    for T in (Float64, ComplexF64)
        t1 = randn(T, W1 ⊗ W2, W3 ⊗ W4)
        t2 = randn(T, W5, W1')
        tc1 = copy(t1)
        tk = kron(t1, t2)
        @test t1 == tc1 # kron does not modify its input
        @test space(tk) == ((codomain(t1) ⊗ codomain(t2)) ← (domain(t1) ⊗ domain(t2)))
        # dense comparison: leg-concatenated Kronecker product (cf. Deligne test)
        a1 = convert(Array, t1)
        a2 = convert(Array, t2)
        ak = convert(Array, tk)
        sz1 = size(a1)
        sz2 = size(a2)
        b1 = reshape(a1, (sz1[1], sz1[2], 1, sz1[3], sz1[4], 1))
        b2 = reshape(a2, (1, 1, sz2[1], 1, 1, sz2[2]))
        expected = b1 .* b2
        @test reshape(ak, size(expected)) ≈ expected
    end
end
