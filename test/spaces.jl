println("------------------------------------")
println("|     Fields and vector spaces     |")
println("------------------------------------")
@timedtestset "Fields and vector spaces" verbose = true begin
    @timedtestset "fuse" begin
        for (V1, V2, V3, V4, V5) in (VZ2,)
            # fuse accepts tuples, including the empty tuple (unit object)
            @test @constinferred fuse(()) == oneunit(Z2Space)
            @test fuse((V1,)) == V1
            @test fuse((V1, V2)) == fuse(V1, V2)
            @test fuse((V1, V2, V3)) == fuse(fuse(V1, V2), V3)
            @test fuse((V1, V1')) == Z2Space(0 => 2, 1 => 2)
            @test fuse(ntuple(_ -> zero(Z2Space), 3)) == zero(Z2Space)
        end
    end
    @timedtestset "HomSpace" begin
        for (V1, V2, V3, V4, V5) in (VZ2,)
            W = TK.HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
            @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
            @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
            # @test eval(Meta.parse(sprint(show, W))) == W
            # @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
            @test spacetype(W) == typeof(V1)
            @test sectortype(W) == sectortype(V1)
            @test W[1] == V1
            @test W[2] == V2
            @test W[3] == V3'
            @test W[4] == V4'
            @test W[5] == V5'
            @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
            @test W == deepcopy(W)
            @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
            @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
            @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TK.compose(W, W')
        end
    end
    @timedtestset "insert/remove unit spaces" begin
        V1, V2, V3, V4, V5 = VZ2
        u = oneunit(Z2Space)
        ud = u'
        P = V1 ⊗ V2 ⊗ V3
        @test insertleftunit(P) == (V1 ⊗ V2 ⊗ V3 ⊗ u)
        @test insertleftunit(P, 2) == (V1 ⊗ u ⊗ V2 ⊗ V3)
        @test insertleftunit(P, 4; dual = true) == (V1 ⊗ V2 ⊗ V3 ⊗ ud)
        @test insertrightunit(P) == (V1 ⊗ V2 ⊗ V3 ⊗ u)
        @test insertrightunit(P, 0) == (u ⊗ V1 ⊗ V2 ⊗ V3)
        @test insertrightunit(P, 2; dual = true) == (V1 ⊗ V2 ⊗ ud ⊗ V3)
        @test removeunit(insertleftunit(P, 2), 2) == P
        @test removeunit(insertrightunit(P, 1), 2) == P
        @test_throws ArgumentError removeunit(P, 2)
        @test_throws ArgumentError insertleftunit(P, 5)
        @test_throws ArgumentError insertrightunit(P, 5)

        W = V1 ⊗ V2 ← V3 ⊗ V4
        @test insertleftunit(W) == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ u)
        @test insertleftunit(W, 1) == (u ⊗ V1 ⊗ V2 ← V3 ⊗ V4)
        @test insertleftunit(W, 3) == (V1 ⊗ V2 ← u ⊗ V3 ⊗ V4)
        @test insertleftunit(W, 4) == (V1 ⊗ V2 ← V3 ⊗ u ⊗ V4)
        @test insertleftunit(W, 5; dual = true) == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ ud)
        @test insertrightunit(W) == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ u)
        @test insertrightunit(W, 0) == (u ⊗ V1 ⊗ V2 ← V3 ⊗ V4)
        @test insertrightunit(W, 2; dual = true) == (V1 ⊗ V2 ⊗ ud ← V3 ⊗ V4)
        @test insertrightunit(W, 3) == (V1 ⊗ V2 ← V3 ⊗ u ⊗ V4)
        @test removeunit(insertleftunit(W, 3), 3) == W
        @test removeunit(insertrightunit(W, 3), 4) == W
        @test removeunit(insertrightunit(W, 0), 1) == W
        @test_throws ArgumentError removeunit(W, 1)
        @test_throws ArgumentError removeunit(W, 4)
    end
end
