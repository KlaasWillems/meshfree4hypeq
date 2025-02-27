module Meshfree4ScalarEq

# Imports
using Random

# Project wide random generator object
global const SEED = 10
global const rng = MersenneTwister(SEED)

# Include submodules
include("ScalarHyperbolicEquations.jl")
using .ScalarHyperbolicEquations

include("FluxFunctions.jl")
using .FluxFunctions

include("Particles.jl")
using .Particles

include("SimSettings.jl")
using .SimSettings

include("ParticleGrids.jl")
using .ParticleGrids

include("Interpolations.jl")
using .Interpolations

include("TimeIntegration.jl")
using .TimeIntegration

include("ParticleGridStability.jl")
using .ParticleGridStability

end  # module 