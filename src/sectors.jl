

abstract type Sector end

struct Z2Irrep <: Sector
    n::UInt8
    function Z2Irrep(n::Integer)
        return new(UInt8(mod(n, 2)))
    end
end

Z2Irrep() = Z2Irrep(0)



# Nsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = c == first(a ⊗ b)
Nsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = c == a ⊗ b
frobenius_schur_phase(a::Z2Irrep) = 1
Asymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = Int(Nsymbol(a, b, c))

function fusiontensor(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep)
    return fill(Int(Nsymbol(a, b, c)), (1, 1, 1, 1))
end



charge(c::Z2Irrep) = Int(c.n)

Base.convert(::Type{Z2Irrep}, n::Real) = Z2Irrep(n)

unit(::Type{Z2Irrep}) = Z2Irrep(zero(UInt8))
dual(c::Z2Irrep) = c # typeof(c)(N - c.n)

⊗(c::Z2Irrep) = c
⊗(c::Z2Irrep, cs::Vararg{Z2Irrep}) = Z2Irrep(sum(c.n for c in cs) + c.n)


Base.hash(c::Z2Irrep, h::UInt) = hash(c.n, h)
Base.isless(c1::Z2Irrep, c2::Z2Irrep) = isless(c1.n, c2.n)

unit(a::Z2Irrep) = unit(typeof(a))
Base.one(a::Z2Irrep) = unit(a)
Base.one(::Type{Z2Irrep}) = unit(Z2Irrep)

function isunit(a::Z2Irrep)
    a == unit(a)
end
Base.isone(a::Z2Irrep) = isunit(a)

Base.conj(a::Z2Irrep) = dual(a)

dim(a::Z2Irrep) = 1
