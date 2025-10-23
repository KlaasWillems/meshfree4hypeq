module Interpolations

using LinearAlgebra
using Statistics
using ..Meshfree4ScalarEq.ParticleGrids
using ..Meshfree4ScalarEq.SimSettings

export functionInterpolation!, gradInterpolation!, setCurvatures!, GradientInterpolator, initTimeStep, UpwindGradient, CentralGradient, WENO, MUSCL, AxelMUSCL, DumbserWENO, MLSWeightFunction, inverseWeightFunction, exponentialWeightFunction, getStencil

# Weightfunction logic
abstract type MLSWeightFunction end
struct exponentialWeightFunction <: MLSWeightFunction end
struct inverseWeightFunction <: MLSWeightFunction end

@inline function (w::exponentialWeightFunction)(dxVec; param::Real, normalisation::Real)
    return @. exp(-param*((dxVec/normalisation)^2))
end

@inline function (w::exponentialWeightFunction)(dxVec, dyVec; param::Real, normalisation::Real)
    return @. exp(-param*((dxVec^2 + dyVec^2)/(normalisation^2)))
end

@inline function (w::inverseWeightFunction)(dxVec; param::Real, normalisation::Real)
    return @. 1/(dxVec^2)
end

@inline function (w::inverseWeightFunction)(dxVec, dyVec; param::Real, normalisation::Real)
    return @. 1/(dxVec^2 + dyVec^2)
end

"""
    functionInterpolation!(dxVec::AV1, wVec::AV2, fVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
Polynomial interpolation of the function based on Taylor-Polynomial least squares method. The LS system is solved by solving the normal equations. In case of bad conditioning, scale dxVec or solve LS system using QR method.

wVec and res are overwritten.

# Arguments:
- `dxVec::AbstractVector`: Vector that contains x(j)-x(i) with x(i) the point at which we want to commpute the function value and x(j) the surrounding interpolation points.
- `wVec::AbstractVector`: Weight vector in Least squares problem. 
- `fVec::AbstractVector`: Function values at points x(j).
- `res::AbstractVector`: Vector that will contains the result. res[1] is always the interpolated value, res[2] the first order derivative at x(i), res[3] the second order derivative ... Derivatives are only returned if the order is high enough.
- `order::Int64`: Order of polynomial approximation.
"""
function functionInterpolation!(dxVec::AV1, wVec::AV2, fVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
    @assert length(dxVec) == length(wVec)
    @assert length(wVec) == length(fVec)
    @assert length(res) >= order + 1
    @assert length(wVec) >= order + 1 "At least $(order+1) points need for order $(order) LS interpolation. Only $(length(wVec)) points given."
    @assert !any(isnan, fVec)
    @assert !any(isnan, dxVec)

    if order == 0
        res[1] = dot(wVec, fVec) / sum(wVec)
    elseif order == 1
        b1 = dot(fVec, wVec)
        A11 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b2 = dot(fVec, wVec)
        A12 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A22 = sum(wVec)
        res[1] = (b2 - A22*b1/A12)/(A12 - A22*A11/A12)
        res[2] = (b1 - A11*res[1])/A12
    elseif order == 2
        # Generate normal equations
        b1 = dot(wVec, fVec)
        A11 = sum(wVec)
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        A12 = sum(wVec)
        b2 = dot(wVec, fVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A22 = sum(wVec)
        A13 = A22/2
        b3 = dot(wVec, fVec)/2
        wVec .= wVec .* dxVec  # wVec = dx.^3 .* wVec
        A23 = sum(wVec)/2
        A33 = dot(wVec, dxVec)/4

        # Hardcoded solve of 3x3 LU method
        L21 = A12/A11
        L31 = A13/A11
        U22 = A22-L21*A12
        L32 = (A23-L31*A12)/U22
        U23 = A23-L21*A13
        U33 = A33 - L31*A13 - L32*U23
        y2 = b2 - L21*b1
        y3 = b3 - L31*b1 - L32.*y2

        res[3] = y3 / U33
        res[2] = (y2 - U23*res[3]) / U22
        res[1] = (b1 - A12*res[2] - A13*res[3]) / A11
    else
        error("Order not implemented.")
    end
    @assert !any(isnan, res) "Function contains NaN's in functionInterpolation! method."
end

"""
    gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
Same as `functionInterpolation!` but interpolation for the gradient. dfVec now contains (f(x(j))-f(x(i)), ...)

"""
function gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
    @assert length(dxVec) == length(wVec)
    @assert length(wVec) == length(dfVec)
    @assert length(res) >= order
    @assert length(wVec) >= order "At least $(order) points need for order $(order) LS interpolation. Only $(length(wVec)) points given."
    @assert !any(isnan, dfVec)
    @assert !any(isnan, dxVec)

    if order == 1
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b1 = dot(dfVec, wVec)
        @assert !isnan(b1) "$(b1), $(dfVec), $(wVec)"
        A11 = dot(wVec, dxVec)
        res[1] = b1/A11
        @assert !any(isnan, res[1]) "Gradient contains NaN's in gradInterpolation! method. $(res), $(A11), $(dfVec), $(wVec), $(dxVec)"
        @assert !any(isinf, res[1]) "Gradient contains Inf's in gradInterpolation! method. $(res), $(A11), $(dfVec), $(wVec), $(dxVec)"
    elseif order == 2
        # Generate normal equations
        wVec .= wVec .* dxVec  # wVec = dx .* wVec
        b2 = dot(wVec, dfVec)
        wVec .= wVec .* dxVec  # wVec = dx.^2 .* wVec
        A11 = sum(wVec)
        b3 = dot(wVec, dfVec)/2
        wVec .= wVec .* dxVec  # wVec = dx.^3 .* wVec
        A12 = sum(wVec)/2
        A22 = dot(wVec, dxVec)/4

        # Explicit solve of 2x2 linear system
        res[1] = (b3*A12 - A22*b2)/((A12^2) - A22*A11)
        res[2] = (b3 - A12*res[1])/A22
        @assert !any(isnan, res[1:2]) "Gradient contains NaN's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec)"
        @assert !any(isinf, res[1:2]) "Gradient contains Inf's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec)"
    else
        error("Order not implemented.")
    end
end

"""
    gradInterpolation!(dxVec::AV1, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=2) where {AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real}}
2D Gradient interpolation. At least 2 points required for order 1 (res should of length 2). At least 5 points required for order 2 (res should be of length five).
"""
function gradInterpolation!(dxVec::AV1, dyVec::AV5, wVec::AV2, dfVec::AV3, res::AV4; order::Int64=1) where {    AV1 <: AbstractVector{<:Real}, AV2 <: AbstractVector{<:Real},
                                                                                                                AV3 <: AbstractVector{<:Real}, AV4 <: AbstractVector{<:Real},
                                                                                                                AV5 <: AbstractVector{<:Real}}
    @assert length(dxVec) == length(wVec) == length(dyVec)
    @assert length(wVec) == length(dfVec)
    @assert !any(isnan, dfVec)
    @assert !any(isnan, dxVec)
    @assert !any(isnan, dyVec)

    if order == 1
        @assert length(res) == 2
        # Create 2x2 linear system
        A11 = A12 = A22 = b1 = b2 = 0.0
        for (w, dx) in zip(wVec, dxVec)
            A11 += w*(dx^2)
        end
        for (w, dy) in zip(wVec, dyVec)
            A22 += w*(dy^2)
        end
        for (w, dx, dy) in zip(wVec, dxVec, dyVec)
            A12 += w*dx*dy
        end
        for (w, dx, df) in zip(wVec, dxVec, dfVec)
            b1 += w*dx*df
        end
        for (w, dy, df) in zip(wVec, dyVec, dfVec)
            b2 += w*dy*df
        end
        # Explicit solve of 2x2 linear system
        res[1] = (b2*A12 - A22*b1)/((A12^2) - A22*A11)
        res[2] = (b2 - A12*res[1])/A22
        @assert !any(isnan, res) "Gradient contains NaN's in gradInterpolation! method. $(res), $((A12^2) - A22*A11), $(dfVec), $(dxVec), $(dyVec)"
    elseif order == 2
        @assert length(res) == 5
        # Solve LS problem using Julia's backslash operator. Requires an allocation (A), but condition number doesn't square!
        A = Matrix{Float64}(undef, length(dxVec), 5)
        @. A[:, 1] = dxVec * wVec
        @. A[:, 2] = dyVec * wVec
        @. A[:, 3] = (dxVec^2) * wVec / 2
        @. A[:, 4] = (dyVec^2) * wVec / 2
        @. A[:, 5] = dxVec * dyVec * wVec
        wVec .= wVec .* dfVec
        res .= A \ wVec
    else
        error("Order not implemented.")
    end
end

"""
    GradientInterpolator

In case of unstructured grids, the spatial gradient is approximated using a moving least squares (MLS) method based on Taylor polynomials.
These algorithms are implemented as follows. Each method is a struct that is a subtype of GradientInterpolator. The gradient 
at a gridpoint can then be computed using the ()-operator; see for example UpwindGradient and CentralGradient. These objects select
the correct stencil and then call the MLS routine (gradInterpolation).
"""
abstract type GradientInterpolator end

function initTimeStep(g::GradientInterpolator, particleGrid::ParticleGrid, interpAlpha::Real, interpRange::Real) end  # Function called at the start of a time step (order RK-stage)

# ------------------------------- Upwind -------------------------------
abstract type UpwindAlgorithm end  # Only relevant in 2D. In 1D, all algorithms are the same.
abstract type TiwariAlgorithm <: UpwindAlgorithm end  # Split domain in left and right for d/dx, and up and down for d/dy.
abstract type PraveenAlgorithm <: UpwindAlgorithm end  # Praveen C. postive upwind scheme.
abstract type ClassicAlgorithm <: UpwindAlgorithm end  # Take all points 'behind' center point. 
struct UpwindGradient{Algorithm} <: GradientInterpolator where {Algorithm <: UpwindAlgorithm}
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction

    """
        UpwindGradient(order::Int64 = 1; algType::String = "")

    Constructor for Upwind Object. algType only has impact in 2D upwinding.
    """
    function UpwindGradient(order::Int64 = 1; algType::String = "Classic", weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        @assert algType in ["Classic", "Tiwari", "Praveen"]
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        if algType == "Classic"
            new{ClassicAlgorithm}(order, Vector{Float64}(undef, size), weightFunction)
        elseif algType == "Praveen"
            @assert order == 1
            new{PraveenAlgorithm}(order, Vector{Float64}(undef, size), weightFunction)
        elseif algType == "Tiwari"
            new{TiwariAlgorithm}(order, Vector{Float64}(undef, size), weightFunction)
        end
    end
end

function (upwind::UpwindGradient)(particleGrid::ParticleGrid1D{BF}, particleIndex::Integer, fVec::Vector{<:Real}, vel::Real, settings::SimSetting; setCurvature::Bool=true)::Real where {BF}
    dxVec = Vector{Float64}(undef, 0)
    dfVec = Vector{Float64}(undef, 0)
    for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
        deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        if ((vel >= 0.0) && (deltaPos <= 0.0)) || ((vel <= 0.0) && (deltaPos >= 0.0))
            push!(dxVec, deltaPos/settings.interpRange)
            push!(dfVec, fVec[nbIndex] - fVec[particleIndex])
        end
    end
    wVec = upwind.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
    @assert !any(isnan, wVec) && !any(isinf, wVec) "Infs or Nan's in wVec: $(wVec)"

    if !((BF == 1) || (BF == 2) && (particleIndex == 1 || particleIndex == length(particleGrid.grid)))
        gradInterpolation!(dxVec, wVec, dfVec, upwind.res; order=upwind.order)
        if setCurvature
            particleGrid.grid[particleIndex].curvature = 0.0
        end
        return vel*upwind.res[1]/settings.interpRange
    else
        return 0.0
    end

end

function (upwind::UpwindGradient{TiwariAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real    
    nbNeighbours = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, nbNeighbours)
    dyVec = Vector{Float64}(undef, nbNeighbours)
    dfVec = Vector{Float64}(undef, nbNeighbours)    
    xWindow = Vector{Bool}(undef, nbNeighbours)  # True if point should be used for d/dx
    yWindow = Vector{Bool}(undef, nbNeighbours)  # True if point should be used for d/dx

    for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/settings.interpRange
        dyVec[i] = deltaY/settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        xWindow[i] = ((vel[1] >= 0.0) && (deltaX <= 0.0)) || ((vel[1] <= 0.0) && (deltaX >= 0.0))
        yWindow[i] = ((vel[2] >= 0.0) && (deltaY <= 0.0)) || ((vel[2] <= 0.0) && (deltaY >= 0.0))
    end
    wVec = upwind.weightFunction(dxVec[xWindow], dyVec[xWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[xWindow], dyVec[xWindow], wVec, dfVec[xWindow], upwind.res; order=upwind.order)
    ddx = upwind.res[1]/settings.interpRange

    if upwind.order == 1  # Set the curvature
        particleGrid.grid[particleIndex].curvature[1] = 0.0
    elseif upwind.order == 2
        particleGrid.grid[particleIndex].curvature[1] = upwind.res[3]/(settings.interpRange^2)
    end
        
    wVec = upwind.weightFunction(dxVec[yWindow], dyVec[yWindow]; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec[yWindow], dyVec[yWindow], wVec, dfVec[yWindow], upwind.res; order=upwind.order)
    ddy = upwind.res[2]/settings.interpRange

    if setCurvature && (upwind.order == 1)
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (upwind.order == 2)
        particleGrid.grid[particleIndex].curvature[2] = upwind.res[4]/(settings.interpRange^2)
    end

    return ddx*vel[1] + ddy*vel[2]
end

function (upwind::UpwindGradient{ClassicAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real    
    dxVec = Vector{Float64}(undef, 0)
    dyVec = Vector{Float64}(undef, 0)
    dfVec = Vector{Float64}(undef, 0)    
    for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        if deltaX*vel[1] + deltaY*vel[2] < 0
            push!(dxVec, deltaX/settings.interpRange)
            push!(dyVec, deltaY/settings.interpRange)
            push!(dfVec, fVec[nbIndex] - fVec[particleIndex])
        end
    end
    wVec = upwind.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, upwind.res; order=upwind.order)

    if setCurvature && (upwind.order == 1) 
        particleGrid.grid[particleIndex].curvature[1] = 0.0
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (upwind.order == 2)
        particleGrid.grid[particleIndex].curvature[1] = upwind.res[3]/(settings.interpRange^2)
        particleGrid.grid[particleIndex].curvature[2] = upwind.res[4]/(settings.interpRange^2)
    end
    
    return vel[1]*upwind.res[1]/settings.interpRange + vel[2]*upwind.res[2]/settings.interpRange
end

function (upwind::UpwindGradient{PraveenAlgorithm})(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real    
    particle = particleGrid.grid[particleIndex]

    if setCurvature
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
    end
    
    # Create 2x2 LS system
    A11 = A12 = A22 = 0.0
    for nbIndex in particle.neighbourIndices
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        A11 += w*(deltaX^2) 
        A12 += w*deltaX*deltaY
        A22 += w*(deltaY^2)
    end
    D = A11*A22 - (A12^2)
    
    div = 0.0
    for nbIndex in particle.neighbourIndices

        # Solve 2x2 LS system
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        w = upwind.weightFunction(deltaX, deltaY; param=settings.interpAlpha, normalisation=settings.interpRange)
        coeff = ((A22*w*deltaX - A12*w*deltaY)/D, (A11*w*deltaY - A12*w*deltaX)/D)

        # Compute adapted coefficients
        angle = atan(deltaY, deltaX)  # atan2 function
        n = (cos(angle), sin(angle))
        s = (-sin(angle), cos(angle))
        alfaBar = dot(n, coeff)
        betaBar = dot(s, coeff)
        bracketMinus = dot(vel, n) > 0.0 ? 0.0 : dot(vel, n)
        bracketMinus2 = betaBar*dot(vel, s) > 0.0 ? 0.0 : betaBar*dot(vel, s)
        cij = alfaBar*bracketMinus + bracketMinus2
        div += 2*cij*(fVec[nbIndex] - fVec[particleIndex])
    end
    return div
end

# ------------------------------- CentralGradient -------------------------------
struct CentralGradient <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction

    function CentralGradient(order::Int64 = 1; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 1 "Order must be larger or equal to one."
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        new(order, Vector{Float64}(undef, size), weightFunction)
    end
end

function (central::CentralGradient)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::AbstractArray{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dyVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)

    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/particleGrid.dx
        dyVec[i] = deltaY/particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    wVec = central.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, dyVec, wVec, dfVec, central.res; order=central.order)

    if setCurvature && (central.order == 1)
        particleGrid.grid[particleIndex].curvature[1] = 0.0
        particleGrid.grid[particleIndex].curvature[2] = 0.0
    elseif setCurvature && (central.order == 2)
        particleGrid.grid[particleIndex].curvature[1] = central.res[3]/(particleGrid.dx^2)
        particleGrid.grid[particleIndex].curvature[2] = central.res[4]/(particleGrid.dx^2)
    end

    return vel[1]*central.res[1]/particleGrid.dx + vel[2]*central.res[2]/particleGrid.dx
end

function (central::CentralGradient)(particleGrid::ParticleGrid1D{BF}, particleIndex::Integer, fVec::Vector{<:Real}, vel::Real, settings::SimSetting; setCurvature::Bool=true)::Real where {BF}
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)

    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaPos/particleGrid.dx
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
    end
    wVec = central.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    gradInterpolation!(dxVec, wVec, dfVec, central.res; order=central.order)

    if setCurvature && (central.order == 1)
        particleGrid.grid[particleIndex].curvature = 0.0
    elseif setCurvature && (central.order == 2)
        particleGrid.grid[particleIndex].curvature = central.res[2]/(particleGrid.dx^2)
    end

    return vel*central.res[1]/particleGrid.dx
end

# ------------------------------- WENO -------------------------------
struct WENO <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction

    function WENO(order::Int64 = 1; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order >= 2 "Order must be larger or equal to two, since the WENO weights require a second derivative."
        if order == 1
            size = 2  # In 2D res has length 2, in 1D res has length 1
        elseif order == 2
            size = 5  # In 2D res had length 5, in 1D res has length 2
        end
        new(order, Vector{Float64}(undef, size), weightFunction)
    end
end

function (weno::WENO)(particleGrid::ParticleGrid1D{BF}, particleIndex::Integer, fVec::Vector{<:Real}, vel::Real, settings::SimSetting; setCurvature::Bool=true)::Real where {BF}
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)
    leftWindow = Vector{Bool}(undef, Npts)
    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaPos
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = deltaPos > 0.0 ? false : true
    end
    wVec = weno.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)

    # One-sided stencil
    if vel > 0.0
        # Left stencil
        if sum(leftWindow) > 1
            gradInterpolation!(dxVec[leftWindow], wVec[leftWindow], dfVec[leftWindow], weno.res; order=weno.order)
        else
            weno.res[1] = 10000.0 # force values of derivatives to be large so that the central stencil is chosen
            weno.res[2] = 10000.0
        end

    else
        # Right stencil
        if sum(.!leftWindow) > 1
            gradInterpolation!(dxVec[.!leftWindow], wVec[.!leftWindow], dfVec[.!leftWindow], weno.res; order=weno.order)
        else
            weno.res[1] = 10000.0
            weno.res[2] = 10000.0
        end
    end
    resS1 = weno.res[1]
    resS2 = weno.res[2]

    # Central stencil
    wVec .= weno.weightFunction(dxVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, wVec, dfVec, weno.res; order=weno.order)
    resC1 = weno.res[1]
    resC2 = weno.res[2]

    e = 1e-6
    dx2 = particleGrid.dx^2
    dx4 = dx2^2
    betaS = 0.5/(((resS1^2)*dx2 + (resS2^2)*dx4 + e)^2)
    betaC = 0.5/(((resC1^2)*dx2 + (resC2^2)*dx4 + e)^2)
    ω_s = betaS/(betaC + betaS)
    ω_c = betaC/(betaC + betaS)

    if setCurvature
        particleGrid.grid[particleIndex].curvature = resS2*ω_s + resC2*ω_c
    end

    res = resS1*ω_s + resC1*ω_c
    @assert !isnan(res) "$(weno.res), $(weno.res), $(betaS), $(betaC), $(ω_s), $(ω_c), $(dfVec)"
    return res*vel
end

function (weno::WENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real
    Npts = length(particleGrid.grid[particleIndex].neighbourIndices)
    dxVec = Vector{Float64}(undef, Npts)
    dyVec = Vector{Float64}(undef, Npts)
    dfVec = Vector{Float64}(undef, Npts)
    leftWindow = Vector{Bool}(undef, Npts)
    topWindow = Vector{Bool}(undef, Npts)
    for i in eachindex(particleGrid.grid[particleIndex].neighbourIndices)
        nbIndex = particleGrid.grid[particleIndex].neighbourIndices[i]
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        dxVec[i] = deltaX/settings.interpRange
        dyVec[i] = deltaY/settings.interpRange
        dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        leftWindow[i] = deltaX > 0.0 ? false : true
        topWindow[i] = deltaY > 0.0 ? true : false
    end
    wVec = weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)

    # One-sided stencil - Left & Right
    if vel[1] > 0.0
        # Left stencil
        gradInterpolation!(dxVec[leftWindow], dyVec[leftWindow], wVec[leftWindow], dfVec[leftWindow], weno.res; order=weno.order)
    else
        # Right stencil
        gradInterpolation!(dxVec[.!leftWindow], dyVec[.!leftWindow], wVec[.!leftWindow], dfVec[.!leftWindow], weno.res; order=weno.order)
    end
    resHx = weno.res[1]/settings.interpRange
    resHy = weno.res[2]/settings.interpRange
    resHxx = weno.res[3]/(settings.interpRange^2)
    resHyy = weno.res[4]/(settings.interpRange^2)
    resHxy = weno.res[5]/(settings.interpRange^2)
    
    # One-sided stencil - Up & Down
    wVec .= weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    if vel[2] < 0.0
        # Top stencil
        gradInterpolation!(dxVec[topWindow], dyVec[topWindow], wVec[topWindow], dfVec[topWindow], weno.res; order=weno.order)
    else
        # Bottom stencil
        gradInterpolation!(dxVec[.!topWindow], dyVec[.!topWindow], wVec[.!topWindow], dfVec[.!topWindow], weno.res; order=weno.order)
    end
    resVx = weno.res[1]/settings.interpRange
    resVy = weno.res[2]/settings.interpRange
    resVxx = weno.res[3]/(settings.interpRange^2)
    resVyy = weno.res[4]/(settings.interpRange^2)
    resVxy = weno.res[5]/(settings.interpRange^2)

    # Central stencil
    wVec .= weno.weightFunction(dxVec, dyVec; param=settings.interpAlpha, normalisation=1.0)
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, weno.res; order=weno.order)
    resCx = weno.res[1]/settings.interpRange
    resCy = weno.res[2]/settings.interpRange
    resCxx = weno.res[3]/(settings.interpRange^2)
    resCyy = weno.res[4]/(settings.interpRange^2)
    resCxy = weno.res[5]/(settings.interpRange^2)

    # Compute non-linear weights
    e = 1e-12
    dx2 = particleGrid.dx^2
    dx4 = dx2^2
    betaH = 0.5/((resHx^2)*dx2 + (resHy^2)*dx2 + (resHxx^2)*dx4 + (resHyy^2)*dx4 + (resHxy^2)*dx4 + e)^2
    betaV = 0.5/((resVx^2)*dx2 + (resVy^2)*dx2 + (resVxx^2)*dx4 + (resVyy^2)*dx4 + (resVxy^2)*dx4 + e)^2
    betaC = 0.5/((resCx^2)*dx2 + (resCy^2)*dx2 + (resCxx^2)*dx4 + (resCyy^2)*dx4 + (resCxy^2)*dx4 + e)^2
    wH = betaH/(betaH + betaC)
    wCx = betaC/(betaH + betaC)
    wV = betaV/(betaC + betaV)
    wCy = betaC/(betaC + betaV)

    if setCurvature
        particleGrid.grid[particleIndex].curvature[1] = wH*resHxx + wCx*resCxx
        particleGrid.grid[particleIndex].curvature[2] = wV*resVyy + wCy*resCyy
    end

    return (wH*resHx + wCx*resCx)*vel[1] + (wV*resVy + wCy*resCy)*vel[2]
end


# ------------------------------- Dumbser WENO -------------------------------

function getStencil(deltaX::Real, deltaY::Real, s::Int64)
    stencil = convert(Int64, div(s*(atan(deltaY, deltaX) + pi)*4/pi, s))
    stencil = stencil == 8 ? 0 : stencil  # Negative x-axis should be contained in stencil 0
    return stencil
end

struct DumbserWENO <: GradientInterpolator
    order::Int64
    res::Vector{Float64}
    weightFunction::MLSWeightFunction
    s::Integer  # amount of one-sided stencils
    gradients::Matrix{Float64}
    weights::Vector{Float64}

    function DumbserWENO(order::Int64 = 2; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert order == 2 "Order must be to two, since the WENO weights require a second derivative."
        new(order, Vector{Float64}(undef, 5), weightFunction, 8, Matrix{Float64}(undef, (5, 9)), Vector{Float64}(undef, 9))
    end
end

function (weno::DumbserWENO)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real
    @assert settings.interpRange >= sqrt(5.0^2 + 3.0^2)*particleGrid.dx "Interpolation must be sufficiently larger, otherwise one cannot guarantee sufficient neighbours are found." 
    particle = particleGrid.grid[particleIndex]
    Npts = length(particle.neighbourIndices)

    # Divide points in stencils
    windowMatrix = zeros(Bool, (Npts, weno.s+1))
    windowMatrix[:, 1] .= true  # First column is the central stencil

    for i in eachindex(particle.neighbourIndices)
        nbIndex = particle.neighbourIndices[i]
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        particle.dxVec[i] = deltaX/settings.interpRange
        particle.dyVec[i] = deltaY/settings.interpRange
        particle.dfVec[i] = fVec[nbIndex] - fVec[particleIndex]
        stencil = getStencil(deltaX, deltaY, weno.s)  # in [0, 7]
        windowMatrix[i, stencil+2] = true
    end
    for stencil in 1:weno.s+1
        particle.wVec .= weno.weightFunction(particle.dxVec, particle.dyVec; param=settings.interpAlpha, normalisation=1.0)
        
        # There should be at least 5 points in each stencil!
        @assert count(windowMatrix[:, stencil]) >= 5 "($(particle.pos[1]), $(particle.pos[2])), $(count(windowMatrix[:, stencil])), $(stencil)"
        gradInterpolation!(particle.dxVec[windowMatrix[:, stencil]], particle.dyVec[windowMatrix[:, stencil]], particle.wVec[windowMatrix[:, stencil]], particle.dfVec[windowMatrix[:, stencil]], weno.res; order=weno.order)

        # Rescale results
        weno.gradients[1, stencil] = weno.res[1]/settings.interpRange
        weno.gradients[2, stencil] = weno.res[2]/settings.interpRange  
        weno.gradients[3, stencil] = weno.res[3]/(settings.interpRange^2)
        weno.gradients[4, stencil] = weno.res[4]/(settings.interpRange^2)
        weno.gradients[5, stencil] = weno.res[5]/(settings.interpRange^2)

        # Compute weights
        r = 4
        eps = 1e-14
        lambda = (stencil == 1) ? 10^5 : 1.0
        weno.weights[stencil] = lambda/((eps + sum((x^2 for x in weno.gradients[:, stencil])))^r)
    end

    # Normalise weights
    weno.weights .= weno.weights ./ sum(weno.weights)
    
    if setCurvature 
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
        for i in eachindex(weno.weights)  # Write out inner product
            particle.curvature[1] += weno.weights[i]*weno.gradients[3, i]
            particle.curvature[2] += weno.weights[i]*weno.gradients[4, i]
        end
    end

    # Compute divergence
    res = 0.0
    
    for i in eachindex(weno.weights)  # Write out inner product
        res += weno.weights[i]*(weno.gradients[1, i]*vel[1] + vel[2]*weno.gradients[2, i])
    end
    return res
end


# ------------------------------- MUSCL -------------------------------
"""
    MUSCL <: GradientInterpolator

Central reconstruction of any order with upwinding.
"""
mutable struct MUSCL <: GradientInterpolator
    order::Int64
    res::Vector{Float64}  # Result of gradient computation
    weightFunction::MLSWeightFunction

    function MUSCL(order::Int64; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        @assert (order == 1) || (order == 2) || (order == 3) || (order == 4) "Order must be one, two, three or four."
        new(order, Vector{Float64}(undef, order), weightFunction)
    end
end

function initTimeStep(muscl::MUSCL, particleGrid::ParticleGrid1D{BF}, interpAlpha::Real, interpRange::Real) where {BF}
    # Compute reconstruction for particle in each neighbourhood and store gradient coefficients
    for (particleIndex, particle) in enumerate(particleGrid.grid)

        for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
            deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            particle.dxVec[i] = deltaPos
        end
        particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)

        if muscl.order == 1
            @. particle.wVec = particle.wVec * particle.dxVec  # wVec .= dx .* wVec
            t = dot(particle.wVec, particle.dxVec)
            @. particle.alfaij = particle.wVec / t  # Derivative
        elseif muscl.order == 2

            # alfa_ij
            @. particle.wVec = particle.wVec * particle.dxVec  # wVec .= dx .* wVec
            t = dot(particle.wVec, particle.dxVec)
            @. particle.alfaij = particle.wVec / t

            # Restore vector
            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)

            # alfa_ijBar and betaij
            @. particle.wVec = particle.wVec * particle.dxVec * particle.dxVec  # wVec = dx.^2 .* wVec
            A11 = sum(particle.wVec)
            particle.wVec .= particle.wVec .* particle.dxVec  # wVec = dx.^3 .* wVec
            A12 = sum(particle.wVec)/2
            A22 = dot(particle.wVec, particle.dxVec)/4
            D = A11*A22 - A12^2

            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=particleGrid.dx)
            for i in eachindex(particle.alfaijBar)
                particle.alfaijBar[i] = (A22*particle.wVec[i]*particle.dxVec[i] - 0.5*A12*particle.wVec[i]*(particle.dxVec[i]^2))/D  # Derivative
                particle.betaij[i] = (0.5*A11*particle.wVec[i]*(particle.dxVec[i]^2) - A12*particle.wVec[i]*particle.dxVec[i])/D  # Second derivative
            end
        elseif muscl.order == 3
            @. particle.A[:, 1] = particle.dxVec * particle.wVec 
            @. particle.A[:, 2] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 3] = (particle.dxVec^3) * particle.wVec / 6

            coeff = pinv(particle.A[:, 1:3]; rtol=sqrt(eps(real(float(oneunit(eltype(particle.A)))))))
            @. particle.alfaijBar = coeff[1, :] * particle.wVec  # Derivative
            @. particle.betaij = coeff[2, :] * particle.wVec  # Second derivative
            @. particle.alfaij = coeff[3, :] * particle.wVec  # Third derivative
        elseif muscl.order == 4
            particle.wVec .= muscl.weightFunction(particle.dxVec; param=interpAlpha, normalisation=1.0)
            @. particle.A[:, 1] = particle.dxVec * particle.wVec 
            @. particle.A[:, 2] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 3] = (particle.dxVec^3) * particle.wVec / 6
            @. particle.A[:, 4] = (particle.dxVec^4) * particle.wVec / 24

            coeff = pinv(particle.A; rtol=sqrt(eps(real(float(oneunit(eltype(particle.A)))))))
            @. particle.alfaijBar = coeff[1, :] * particle.wVec  # Derivative
            @. particle.betaij = coeff[2, :] * particle.wVec  # Second derivative
            @. particle.alfaij = coeff[3, :] * particle.wVec  # Third derivative
            @. particle.gammaij = coeff[4, :] * particle.wVec  # Fourth derivative
        end
    end
end

function (muscl::MUSCL)(particleGrid::ParticleGrid1D{BF}, particleIndex::Integer, fVec::Vector{<:Real}, vel::Real, settings::SimSetting; setCurvature::Bool=true)::Real where{BF}
    particle = particleGrid.grid[particleIndex]
    div = 0.0
    for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        nbParticle = particleGrid.grid[nbIndex]

        if muscl.order == 1
            # Linear reconstruction from particleIndex and neighbour at center point 
            if vel*deltaPos > 0.0
                fij = fVec[particleIndex] + 0.5*deltaPos*sum(particle.alfaij[i]*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
            else
                fij = fVec[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaij[i]*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
            end
            div += particle.alfaij[index]*(fij - fVec[particleIndex])
        elseif muscl.order == 2
            # Quadratic reconstruction
            if vel*deltaPos > 0.0
                fij = fVec[particleIndex]
                for (i, k) in enumerate(particle.neighbourIndices)
                    fij += (deltaPos*particle.alfaijBar[i]/2 + (deltaPos^2)*particle.betaij[i]/8)*(fVec[k] - fVec[particleIndex])
                end
            else
                fij = fVec[nbIndex]
                for (i, k) in enumerate(nbParticle.neighbourIndices)
                    fij += (-deltaPos*nbParticle.alfaijBar[i]/2 + (deltaPos^2)*nbParticle.betaij[i]/8)*(fVec[k] - fVec[nbIndex])
                end
            end
            div += particle.alfaijBar[index]*(fij - fVec[particleIndex])
        elseif muscl.order == 3
            # Cubic reconstruction
            if vel*deltaPos > 0.0
                fij = fVec[particleIndex] + 0.5*deltaPos*sum(particle.alfaijBar[i]*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                fij += (deltaPos^2)*sum(particle.betaij[i]*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/8
                fij += (deltaPos^3)*sum(particle.alfaij[i]*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/(6*8)
            else
                fij = fVec[nbIndex] - 0.5*deltaPos*sum(nbParticle.alfaijBar[i]*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                fij += (deltaPos^2)*sum(nbParticle.betaij[i]*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/8
                fij -= (deltaPos^3)*sum(nbParticle.alfaij[i]*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/(6*8)
            end
            div += particle.alfaijBar[index]*(fij - fVec[particleIndex])
        elseif muscl.order == 4
            # Quartic reconstruction
            if vel*deltaPos > 0.0
                fij = fVec[particleIndex]
                for (i, k) in enumerate(particle.neighbourIndices)
                    fij += (deltaPos*particle.alfaijBar[i]/2 + (deltaPos^2)*particle.betaij[i]/8 + (deltaPos^3)*particle.alfaij[i]/(6*8) + (deltaPos^4)*particle.gammaij[i]/(24*(2^4)))*(fVec[k] - fVec[particleIndex])
                end
            else
                fij = fVec[nbIndex]
                for (i, k) in enumerate(nbParticle.neighbourIndices)
                    fij += (-deltaPos*nbParticle.alfaijBar[i]/2 + (deltaPos^2)*nbParticle.betaij[i]/8 - (deltaPos^3)*nbParticle.alfaij[i]/(6*8) + (deltaPos^4)*nbParticle.gammaij[i]/(24*(2^4)))*(fVec[k] - fVec[nbIndex])
                end
            end
            div += particle.alfaijBar[index]*(fij - fVec[particleIndex])
        end
    end

    # set curvature
    if setCurvature && (muscl.order == 1)
        particle.curvature = 0.0
    elseif setCurvature
        particle.curvature = sum(particle.betaij[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
    end
    return 2*vel*div # Minus sign in front of the divergence taken into account in the time stepper routine
end

function initTimeStep(muscl::MUSCL, particleGrid::ParticleGrid2D, interpAlpha::Real, interpRange::Real)
    # Compute reconstruction for particle in each neighbourhood and store gradient coefficients
    for (particleIndex, particle) in enumerate(particleGrid.grid)

        for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            particle.dxVec[i] = deltaX
            particle.dyVec[i] = deltaY
        end
        particle.wVec .= muscl.weightFunction(particle.dxVec, particle.dyVec; param=interpAlpha, normalisation=interpRange)

        if muscl.order == 1
            A11 = A12 = A22 = 0.0
            for (w, dx) in zip(particle.wVec, particle.dxVec)
                A11 += w*(dx^2)
            end
            for (w, dy) in zip(particle.wVec, particle.dyVec)
                A22 += w*(dy^2)
            end
            for (w, dx, dy) in zip(particle.wVec, particle.dxVec, particle.dyVec)
                A12 += w*dx*dy
            end
            D = (A12^2) - A22*A11

            for i in 1:length(particle.dxVec)
                particle.alfaij[i] = (particle.wVec[i]*particle.dyVec[i]*A12 - A22*particle.wVec[i]*particle.dxVec[i])/D
                particle.betaij[i] = (-particle.wVec[i]*particle.dyVec[i]*A11 + A12*particle.wVec[i]*particle.dxVec[i])/D
            end    
        elseif muscl.order == 2
            @. particle.A[:, 1] = particle.dxVec * particle.wVec
            @. particle.A[:, 2] = particle.dyVec * particle.wVec
            @. particle.A[:, 3] = (particle.dxVec^2) * particle.wVec / 2
            @. particle.A[:, 4] = (particle.dyVec^2) * particle.wVec / 2
            @. particle.A[:, 5] = particle.dxVec * particle.dyVec * particle.wVec

            coeff = pinv(particle.A)
            particle.alfaij .= coeff[1, :] .* particle.wVec
            particle.betaij .= coeff[2, :] .* particle.wVec
            particle.alfaijBar .= coeff[3, :] .* particle.wVec
            particle.betaijBar .= coeff[4, :] .* particle.wVec
            particle.gammaij .= coeff[5, :] .* particle.wVec
        end
    end
end

function (muscl::MUSCL)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting; setCurvature::Bool=true)::Real
    particle = particleGrid.grid[particleIndex]
    div = 0.0
    for (index, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
        deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
        nbParticle = particleGrid.grid[nbIndex]

        if muscl.order == 1  # Linear reconstruction from particleIndex and neighbour at center point 
            if vel[1]*deltaX + vel[2]*deltaY > 0.0
                fij = fVec[particleIndex]
                for (i, k) in enumerate(particle.neighbourIndices)
                    @inbounds fij += (deltaX*particle.alfaij[i] + deltaY*particle.betaij[i])*(fVec[k] - fVec[particleIndex])/2
                end
                # fij = fVec[particleIndex] + sum((deltaX*particle.alfaij[i] + deltaY*particle.betaij[i])*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))/2
            else
                fij = fVec[nbIndex]
                for (i, k) in enumerate(nbParticle.neighbourIndices)
                    @inbounds fij -= (deltaX*nbParticle.alfaij[i] + deltaY*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex])/2
                end
                # fij = fVec[nbIndex] - sum((deltaX*nbParticle.alfaij[i] + deltaY*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))/2
            end
            div += particle.alfaij[index]*(fij - fVec[particleIndex])*vel[1] + particle.betaij[index]*(fij - fVec[particleIndex])*vel[2]
        elseif muscl.order == 2  # Quadratic reconstruction from particleIndex and neighbour at center point
            if deltaX*vel[1] + vel[2]*deltaY > 0.0
                fij = fVec[particleIndex] 
                for (i, k) in enumerate(particle.neighbourIndices)
                    @inbounds fij += (deltaX*particle.alfaij[i] + deltaY*particle.betaij[i])*(fVec[k] - fVec[particleIndex])/2
                    @inbounds fij += ((deltaX^2)*particle.alfaijBar[i]/2 + (deltaY^2)*particle.betaijBar[i]/2 + deltaX*deltaY*particle.gammaij[i])*(fVec[k] - fVec[particleIndex])/4
                end
            else
                fij = fVec[nbIndex]
                for (i, k) in enumerate(nbParticle.neighbourIndices)
                    @inbounds fij -= (deltaX*nbParticle.alfaij[i] + deltaY*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex])/2
                    @inbounds fij += ((deltaX^2)*nbParticle.alfaijBar[i]/2 + (deltaY^2)*nbParticle.betaijBar[i]/2 + deltaX*deltaY*nbParticle.gammaij[i])*(fVec[k] - fVec[nbIndex])/4
                end
            end
            div += particle.alfaij[index]*(fij - fVec[particleIndex])*vel[1] + particle.betaij[index]*(fij - fVec[particleIndex])*vel[2]
        end
    end

    # set curvature
    if setCurvature && (muscl.order == 1)
        particle.curvature[1] = 0.0
        particle.curvature[2] = 0.0
    elseif setCurvature
        particle.curvature[1] = sum(particle.alfaijBar[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
        particle.curvature[1] = sum(particle.betaijBar[i]*(fVec[nbIndex] - fVec[particleIndex]) for (i, nbIndex) in enumerate(particle.neighbourIndices))  # Central difference for second-derivative
    end
    
    return 2*div # Minus sign in front of the divergence taken into account in the time stepper routine
end

# TODO: AxelMUSCL doesn't work. Remove or debug.
"""
    AxelMUSCL <: GradientInterpolator

Second order MUSCL scheme. Reconstruction at some point at constant radius. Divergence at center point computed using Gauss theorem.
"""
mutable struct AxelMUSCL <: GradientInterpolator
    order::Int64
    res::Vector{Float64}  # Result of gradient computation
    weightFunction::MLSWeightFunction

    function AxelMUSCL(order::Int64; weightFunction::MLSWeightFunction = exponentialWeightFunction())
        new(order, Vector{Float64}(undef, order), weightFunction)
    end
end

function initTimeStep(muscl::AxelMUSCL, particleGrid::ParticleGrid2D, interpAlpha::Real, interpRange::Real)
    # Compute reconstruction for particle in each neighbourhood and store gradient coefficients
    for (particleIndex, particle) in enumerate(particleGrid.grid)

        for (i, nbIndex) in enumerate(particle.neighbourIndices)
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            particle.dxVec[i] = deltaX
            particle.dyVec[i] = deltaY
            particle.wVec[i] = sqrt(deltaX^2 + deltaY^2)  # Use temporarily to find smallest distance
            particle.alfaijBar[i] = atan(deltaY, deltaX)  # Store angle!
            if particle.alfaijBar[i] < 0
                particle.alfaijBar[i] += 2*pi
            end
        end

        particle.betaijBar[1] = minimum(particle.wVec)/2  # Store alfa_i (point at which reconstruction takes place)

        # Sort arrays according to increase theta
        p = sortperm(particle.alfaijBar)
        permute!(particle.dxVec, p)
        permute!(particle.dyVec, p)
        permute!(particle.alfaijBar, p)
        permute!(particle.neighbourIndices, p)

        # Solver least squares system
        @. particle.wVec = muscl.weightFunction(particle.dxVec, particle.dyVec; param=interpAlpha, normalisation=particleGrid.dx)

        A11 = A12 = A22 = 0.0
        for (w, dx) in zip(particle.wVec, particle.dxVec)
            A11 += w*(dx^2)
        end
        for (w, dy) in zip(particle.wVec, particle.dyVec)
            A22 += w*(dy^2)
        end
        for (w, dx, dy) in zip(particle.wVec, particle.dxVec, particle.dyVec)
            A12 += w*dx*dy
        end
        D = (A12^2) - A22*A11

        for i in eachindex(particle.alfaij)
            particle.alfaij[i] = (particle.wVec[i]*particle.dyVec[i]*A12 - A22*particle.wVec[i]*particle.dxVec[i])/D  # x-derivative
            particle.betaij[i] = (-particle.wVec[i]*particle.dyVec[i]*A11 + A12*particle.wVec[i]*particle.dxVec[i])/D  # y-derivative
            if i == 1  # Store delta theta's
                particle.gammaij[i] = (particle.alfaijBar[2] + 2*pi - particle.alfaijBar[end])/2
            elseif i == length(particle.alfaijBar)
                particle.gammaij[i] = (particle.alfaijBar[1] + 2*pi - particle.alfaijBar[end-1])/2
            else
                particle.gammaij[i] = (particle.alfaijBar[i+1] - particle.alfaijBar[i-1])/2
            end
        end    
    end
end

function (muscl::AxelMUSCL)(particleGrid::ParticleGrid2D, particleIndex::Integer, fVec::Vector{<:Real}, vel::Tuple{Real, Real}, settings::SimSetting)::Real
    particle = particleGrid.grid[particleIndex]
    div = 0.0
    alfai = particle.betaijBar[1]
    for (index, nbIndex) in enumerate(particle.neighbourIndices)
        nx = particle.dxVec[index]/sqrt(particle.dxVec[index]^2 + particle.dyVec[index]^2)
        ny = particle.dyVec[index]/sqrt(particle.dxVec[index]^2 + particle.dyVec[index]^2)
        
        # Linear reconstruction from particleIndex and neighbour at center point 
        inn = vel[1]*nx + vel[2]*ny
        if inn > 0.0
            fij = fVec[particleIndex] + alfai*sum((nx*particle.alfaij[i] + ny*particle.betaij[i])*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
        else
            nbParticle = particleGrid.grid[nbIndex]
            fij = fVec[nbIndex] + sum(((nx*alfai - particle.dxVec[index])*nbParticle.alfaij[i] + (ny*alfai - particle.dyVec[index])*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
        end
        div += inn*fij*particle.gammaij[index]
    end
    return div/(2*pi)
end

"""
    setCurvatures!(particleGrid::ParticleGrid, settings::SimSetting)

Compute curvatures on whole grid using a central MLS method. Overwrites the particleGrid.temp vector.
"""
function setCurvatures!(particleGrid::ParticleGrid1D{BF}, settings::SimSetting) where {BF}
    central = CentralGradient(2)
    map!(particle -> particle.rho, particleGrid.temp, particleGrid.grid)
    for particleIndex in eachindex(particleGrid.grid)
        central(particleGrid, particleIndex, particleGrid.temp, 0.0, settings)
    end
end

function setCurvatures!(particleGrid::ParticleGrid2D, settings::SimSetting)
    central = CentralGradient(2)
    for particleIndex in eachindex(particleGrid.grid)  # Use first column of particleGrid.temp as temporary
        particleGrid.temp[particleIndex, 1] = particleGrid.grid[particleIndex].rho
    end
    for particleIndex in eachindex(particleGrid.grid)
        central(particleGrid, particleIndex, @view(particleGrid.temp[:, 1]), (0.0, 0.0), settings)
    end
end

end  # module Interpolations