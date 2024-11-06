module Particles

export Particle1D, Particle2D

abstract type AbstractParticle end

"""
    Particle1D

A grid point object. It stores its own position pos, the scalar unknown rho, the indices of its neighbours in ParticleGrid.grid and a boolean array to indicate of those neighbours are to the left (1) or right (0).
"""
mutable struct Particle1D <: AbstractParticle
    pos::Float64
    rho::Float64
    curvature::Float64
    boundary::Bool
    volume::Float64
    neighbourIndices::Vector{Int64}
    alfaij::Vector{Float64}  # Pre-allocated coefficients for interpolation routine
    alfaijBar::Vector{Float64}  
    betaij::Vector{Float64}
    gammaij::Vector{Float64}
    dxVec::Vector{Float64}
    wVec::Vector{Float64}
    dfVec::Vector{Float64}
    A::Matrix{Float64}
    moodEvent::Bool

    function Particle1D(pos::Real, rho::Real, boundary::Bool)
        alfaij = Vector{Float64}(undef, 0)
        alfaijBar = Vector{Float64}(undef, 0)
        betaij = Vector{Float64}(undef, 0)
        gammaij = Vector{Float64}(undef, 0)
        dxVec = Vector{Float64}(undef, 0)
        wVec = Vector{Float64}(undef, 0)
        dfVec = Vector{Float64}(undef, 0)
        new(convert(Float64, pos), convert(Float64, rho), 0.0, boundary, 0.0, Vector{UInt64}(undef, 0), alfaij, alfaijBar, betaij, gammaij, dxVec, wVec, dfVec, Matrix{Float64}(undef, 0, 0), false)
    end
end

"""
    Particle2D

A grid point object. It stores its own position pos, the scalar unknown rho, the indices of its neighbours in ParticleGrid.grid and a boolean array to indicate of those neighbours are to the left (1) or right (0).
"""
mutable struct Particle2D <: AbstractParticle
    pos::Tuple{Float64, Float64}
    rho::Float64
    curvature::Vector{Float64}
    boundary::Bool
    volume::Float64
    voxel::Int64  # Voxel index
    neighbourIndices::Vector{Int64}
    A::Matrix{Float64}
    alfaij::Vector{Float64}  # Pre-allocated coefficients for interpolation routine
    alfaijBar::Vector{Float64}  
    betaij::Vector{Float64}
    betaijBar::Vector{Float64}
    gammaij::Vector{Float64}
    dxVec::Vector{Float64}
    dyVec::Vector{Float64}
    wVec::Vector{Float64}
    dfVec::Vector{Float64}
    moodEvent::Bool

    function Particle2D(pos::Tuple{Real, Real}, rho::Real, boundary::Bool)
        A = Matrix{Float64}(undef, 0, 0)
        alfaij = Vector{Float64}(undef, 0)
        alfaijBar = Vector{Float64}(undef, 0)
        betaij = Vector{Float64}(undef, 0)
        betaijBar = Vector{Float64}(undef, 0)
        gammaij = Vector{Float64}(undef, 0)
        dxVec = Vector{Float64}(undef, 0)
        dyVec = Vector{Float64}(undef, 0)
        wVec = Vector{Float64}(undef, 0)
        dfVec = Vector{Float64}(undef, 0)
        new((convert(Float64, pos[1]), convert(Float64, pos[2])), convert(Float64, rho), [0.0, 0.0], boundary, 0.0, -1, Vector{UInt64}(undef, 0), A, alfaij, alfaijBar, betaij, betaijBar, gammaij, dxVec, dyVec, wVec, dfVec, false)
    end
end

end  # module Particles
