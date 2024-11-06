module ScalarHyperbolicEquations

export ScalarHyperbolicEquation, LinearScalarHyperbolicEquation, NonLinearScalarHyperbolicEquation, LinearAdvection, BurgersEquation, velocity, flux


"""
    ScalarHyperbolicEquation

Each scalar hyperbolic equation is a struct that is a subtype of the abstract type ScalarHyperbolicEquation.
Each struct then overrides the velocity function which evaluates the velocity f'(u, x, t). See LinearAdvection and 
BurgersEquation for examples.
"""
abstract type ScalarHyperbolicEquation end
abstract type LinearScalarHyperbolicEquation <: ScalarHyperbolicEquation end
abstract type NonLinearScalarHyperbolicEquation <: ScalarHyperbolicEquation end

function (eq::ScalarHyperbolicEquation)(rho::Real, x::Real, t::Real)
    error("Not implemented.")
end

struct LinearAdvection{T} <: LinearScalarHyperbolicEquation
    vel::T

    function LinearAdvection(vel::Union{Real, Tuple{<:Real, <:Real}})
        if vel isa Real
            new{Float64}(convert(Float64, vel))
        else
            new{Tuple{Float64, Float64}}((convert(Float64, vel[1]), convert(Float64, vel[2])))
        end
    end
end

function velocity(eq::LinearAdvection, rho::Real, x::Real, t::Real)
    return eq.vel
end

function flux(eq::LinearAdvection, u::Real)
    return eq.vel*u
end

struct BurgersEquation <: NonLinearScalarHyperbolicEquation end

function velocity(eq::BurgersEquation, rho::Real, x::Real, t::Real)
    return rho
end

function flux(eq::BurgersEquation, u::Real)
    return (rho^2)/2
end


end  # module ScalarHyperbolicEquations