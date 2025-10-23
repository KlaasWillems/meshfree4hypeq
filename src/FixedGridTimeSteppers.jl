struct Upwind <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function Upwind(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (upwind::Upwind)(eq::LinearAdvection, particleGrid::ParticleGrid1D{BF}, settings::SimSetting, time::Real, dt::Real) where {BF}
    map!(particle -> particle.rho, upwind.rhoOld, particleGrid.grid)
    vel = velocity(eq, particleGrid.grid[1].rho, particleGrid.grid[1].pos, time)  # Velocity is constant so just evaluate it at the first particle
    λ = vel*dt/particleGrid.dx
    if vel > 0
        particleGrid.grid[1].rho -= λ*(upwind.rhoOld[1] - upwind.rhoOld[end])
        for i in 2:particleGrid.N
            particleGrid.grid[i].rho -= λ*(upwind.rhoOld[i] - upwind.rhoOld[i-1])
        end
    else
        for i in 1:particleGrid.N-1
            particleGrid.grid[i].rho -= λ*(upwind.rhoOld[i+1] - upwind.rhoOld[i])
        end
        particleGrid.grid[end].rho -= λ*(upwind.rhoOld[1] - upwind.rhoOld[end])
    end
end

function (upwind::Upwind)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D{BF}, settings::SimSetting, time::Real, dt::Real) where {BF}
    error("Upwind method for nonlinear hyperbolic equations (Roe's scheme) not yet implemented.")
end

struct LaxFriedrich <: FixedGridTimeStepper 
    rhoOld::Vector{Float64}
    function LaxFriedrich(Nx::Integer)
        new(Vector{Float64}(undef, Nx))
    end
end

function (lf::LaxFriedrich)(eq::ScalarHyperbolicEquation, particleGrid::ParticleGrid1D{BF}, settings::SimSetting, time::Real, dt::Real) where {BF}
    map!(particle -> particle.rho, lf.rhoOld, particleGrid.grid)
    λ = dt/(2*particleGrid.dx)
    for i in 2:particleGrid.N-1
        particleGrid.grid[i].rho = 0.5*(lf.rhoOld[i+1] + lf.rhoOld[i-1]) -λ*(flux(eq, lf.rhoOld[i+1]) - flux(eq, lf.rhoOld[i-1]))
    end
    particleGrid.grid[particleGrid.N].rho = 0.5*(lf.rhoOld[1] + lf.rhoOld[particleGrid.N-1]) -λ*(flux(eq, lf.rhoOld[1]) - flux(eq, lf.rhoOld[particleGrid.N-1]))
    particleGrid.grid[i].rho = 0.5*(lf.rhoOld[i+1] + lf.rhoOld[i-1]) -λ*(flux(eq, lf.rhoOld[i+1]) - flux(eq, lf.rhoOld[i-1]))
end
