module FluxFunctions

using ..Meshfree4ScalarEq.ScalarHyperbolicEquations

export NumericalFluxFunction, RusanovFlux, UpwindFlux

abstract type NumericalFluxFunction end;

# ---------- RusanovFlux (LLF)
struct RusanovFlux <: NumericalFluxFunction end

function (rusanov::RusanovFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation)
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    s = max(abs(velocity(eq, leftState)), abs(velocity(eq, rightState)))
    return 0.5*(leftFlux + rightFlux - s*(rightState - leftState))
end

# ---------- UpwindFlux
struct UpwindFlux <: NumericalFluxFunction end

function (upwind::UpwindFlux)(leftState::Real, rightState::Real, eq::ScalarHyperbolicEquation)
    leftFlux = flux(eq, leftState)
    rightFlux = flux(eq, rightState)
    a = (leftFlux - rightFlux)/(leftState - rightState)
    return 0.5*(leftFlux + rightFlux - a*(rightState - leftState))
end

end