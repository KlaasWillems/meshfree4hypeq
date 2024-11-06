module TimeIntegration

using LinearAlgebra
using ..ParticleGrids
using ..SimSettings
using ..ScalarHyperbolicEquations
using ..Interpolations

export mainTimeIntegrator!

"""
    TimeStepper

Time integration methods are split into two groups: MeshfreeTimeStepper and FixedGridTimeStepper, the former working for all types of grids, 
the latter only working for fixed grid with uniform discretisation parameter. Time integration methods are implemented as structs that override
the ()-operator. See EulerUpwind in "MeshfreeTimeSteppers.jl" as an example. 
"""
abstract type TimeStepper end
abstract type MeshfreeTimeStepper <: TimeStepper end
abstract type FixedGridTimeStepper <: TimeStepper end

function (method::TimeStepper)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting, time::Real, dt::Real)
    error("Each `TimeStepper' must override the ()-operator.")
end

# Function called once before time integration loop to pre-calculate all relevant coefficients fot interpolation.
function initTimeStepper(method::TimeStepper, particleGrid::ParticleGrid, settings::SimSetting) end

include("FixedGridTimeSteppers.jl")
include("MeshfreeTimeSteppers.jl")


"""
    mainTimeIntegrator!

This method performs that actual time integration. Provided with a timeStepper, equation, an initialized grid and simulation settings, it will perform
time integration.
"""
function mainTimeIntegrator!(timeStepper::TimeStepper, eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid, settings::SimSetting)

    if !particleGrid.regular
        @assert timeStepper isa MeshfreeTimeStepper "Must use a MeshfreeTimeStepper for unstructured grids."
    end

    # Initialize grid
    updateNeighbours!(particleGrid, settings.interpRange)
    saveGrid(settings, particleGrid, 0.0)
    setCurvatures!(particleGrid, settings)

    # Initialize interpolation routine
    initTimeStepper(timeStepper, particleGrid, settings)
    t = 0.0
    k = 1
    time = @elapsed while t < settings.tmax
        dt = min(settings.dt, settings.tmax-t)

        # Save data every savefreq steps
        if mod(k, settings.saveFreq) == 0
            saveGrid(settings, particleGrid, t)
        end
        
        timeStepper(eq, particleGrid, settings, t, dt)

        t += dt
        k += 1
    end
    saveGrid(settings, particleGrid, t)
    saveSettings(settings)
    return time
end

end  # module TimeIntegration