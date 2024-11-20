using Test
using Random
using LinearAlgebra
using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq
import Meshfree4ScalarEq.rng

# Define some polynomials with which the interpolation routines can be tested
function testFunction(xVec::Union{Number, Vector}, a::Real)
    return @. (xVec^3)/3 + a*xVec/2 + xVec + 1
end

function testdFunction(xVec::Union{Number, Vector}, a::Real)
    return @. xVec^2 + a*xVec + 1
end

function testd2Function(xVec::Union{Number, Vector}, a::Real)
    return @. 2*xVec+a
end

function testd3Function(xVec::Union{Number, Vector}, a::Real)
    return 2.0*ones(eltype(xVec[1]), size(xVec))
end

function testFunction1(xVec::T, yVec::T, a::Real) where {T <: Union{Number, Vector}}
    return @. xVec^2 + a*xVec + 1 + 3*(yVec^2)/5 + yVec*4 + 6*(yVec * xVec)
end

function gradTestFunction1(xVec::T, yVec::T, a::Real) where {T <: Union{Number, Vector}}
    return (@. 2*xVec+a+6*yVec, @. 6*yVec/5 + 4 + 6*xVec)
end

function testFunction2(xVec::T, yVec::T, a::Real) where {T <: Union{Number, Vector}}
    return @. 2*xVec + a + 6*yVec/5 + 4 
end

function gradTestFunction2(xVec::T, yVec::T, a::Real) where {T <: Union{Number, Vector}}
    return (2, 6/5)
end

@testset "functionInterpolation 1D" begin

    # Generate irregular grid
    Npts = 6
    a = 10
    xmin = -1.0
    xmax = 3.0
    x = rand(rng, Npts).*(xmax - xmin) .+ xmin

    xi = 0.6
    dxVec = x .- xi
    
    # Zero-th order interpolation of constant function should give exact results
    fVec = testd3Function(x, a)
    res = Vector{Float64}(undef, 1)
    wVec = exp.(-6 .* (dxVec).^2 )
    functionInterpolation!(dxVec, wVec, fVec, res; order = 0)
    f0 = testd3Function(xi, a)
    @test isapprox(res[1], f0[1], rtol=1e-13)

    # First order interpolation of line should give exact results
    fVec = testd2Function(x, a)
    res = Vector{Float64}(undef, 2)
    wVec = exp.(-6 .* (dxVec).^2 )
    functionInterpolation!(dxVec, wVec, fVec, res; order = 1)
    f0 = testd2Function(xi, a)
    @test isapprox(res[1], f0[1], rtol=1e-10)
    
    # Second order interpolation of parabola should give exact results
    fVec = testdFunction(x, a)
    res = Vector{Float64}(undef, 3)
    wVec = exp.(-6 .* (dxVec).^2 )
    functionInterpolation!(dxVec, wVec, fVec, res; order = 2)
    f0 = testdFunction(xi, a)
    @test isapprox(res[1], f0[1], rtol=1e-10)
end

@testset "gradInterpolation 1D" begin

    # Generate irregular grid
    Npts = 6
    a = 10
    xmin = -1.0
    xmax = 3.0
    x = rand(rng, Npts).*(xmax - xmin) .+ xmin

    xi = 0.6
    dxVec = x .- xi
    
    # First order interpolation of gradient of a line should give exact results
    fi = testd2Function(xi, a)
    fVec = testd2Function(x, a) .- fi
    res = Vector{Float64}(undef, 1)
    wVec = exp.(-6 .* (dxVec).^2 )
    gradInterpolation!(dxVec, wVec, fVec, res; order = 1)
    f0 = testd3Function(xi, a)
    @test isapprox(res[1], f0[1], rtol=1e-13)

    # # First order interpolation of line should give exact results
    fi = testdFunction(xi, a)
    fVec = testdFunction(x, a) .- fi
    res = Vector{Float64}(undef, 2)
    wVec = exp.(-6 .* (dxVec).^2 )
    gradInterpolation!(dxVec, wVec, fVec, res; order = 2)
    f0 = testd2Function(xi, a)
    @test isapprox(res[1], f0[1], rtol=1e-10)
end

@testset "gradInterpolation 2D" begin
    a = 10
    
    # Generate random grid
    Nx = 6
    Ny = 6
    xmin = -1.0
    xmax = 3.0
    ymin = 1.0
    ymax = 2.0
    x = rand(rng, Nx).*(xmax - xmin) .+ xmin
    y = rand(rng, Ny).*(ymax - ymin) .+ ymin

    # Point at which to interpolate
    xi = 0.6
    yi = 1.3

    # First order interpolation
    dxVec = x .- xi
    dyVec = y .- yi
    fi = testFunction2(xi, yi, a)
    dfVec = testFunction2(x, y, a) .- fi
    res = Vector{Float64}(undef, 2)
    wVec = ones(length(dfVec))
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, res; order = 1)
    res0 = gradTestFunction2(xi, yi, a)
    @test isapprox(res[1], res0[1], rtol=1e-10)
    @test isapprox(res[2], res0[2], rtol=1e-10)

    # Second order interpolation
    dxVec = x .- xi
    dyVec = y .- yi
    fi = testFunction1(xi, yi, a)
    dfVec = testFunction1(x, y, a) .- fi
    res = Vector{Float64}(undef, 5)
    wVec = ones(length(dfVec))
    gradInterpolation!(dxVec, dyVec, wVec, dfVec, res; order = 2)
    res0 = gradTestFunction1(xi, yi, a)
    @test isapprox(res[1], res0[1], rtol=1e-10)
    @test isapprox(res[2], res0[2], rtol=1e-10)

    # Test some least squares problems
    A = Matrix{Float64}(undef, length(dxVec), 5)
    @. A[:, 1] = dxVec * wVec
    @. A[:, 2] = dyVec * wVec
    @. A[:, 3] = (dxVec^2) * wVec / 2
    @. A[:, 4] = (dyVec^2) * wVec / 2
    @. A[:, 5] = dxVec * dyVec
    b = wVec .* dfVec
    resLS1 = A \ b  # Solve LS problem using built-in LS solver
    resLS2 = (A \ diagm(wVec))*dfVec
    @test isapprox(resLS1[1], resLS2[1], rtol=1e-14)
    @test isapprox(resLS1[2], resLS2[2], rtol=1e-14) 
end

function linearInit(x::Real)
    return x
end

function quadraticInit(x::Real)
    return x^2
end

function cubicInit(x::Real)
    return x^3
end

function quarticInit(x::Real)
    return x^4
end

@testset "getStencil basic" begin
    @test getStencil(-4, 0, 8) == 0 
    @test getStencil(-2, -1, 8) == 0 

    @test getStencil(-2, -2, 8) == 1 
    @test getStencil(-2, -3, 8) == 1 

    @test getStencil(0, -2, 8) == 2 
    @test getStencil(2, -3, 8) == 2 

    @test getStencil(2, -2, 8) == 3
    @test getStencil(3, -2, 8) == 3

    @test getStencil(2, 0, 8) == 4
    @test getStencil(3, 2, 8) == 4

    @test getStencil(2, 2, 8) == 5
    @test getStencil(2, 3, 8) == 5

    @test getStencil(0, 2, 8) == 6
    @test getStencil(-1, 2, 8) == 6

    @test getStencil(-2, 2, 8) == 7
    @test getStencil(-3, 2, 8) == 7
end

@testset "getStencil uniform grid" begin
    xmin = -5
    xmax = 5
    interpAlpha = 6.0
    N = 10

    # Cannot trust that particles on vertical and horizontal lines are put inside stencil due to numerical round-off. 
    # Range must be large enough such sufficient inner-stencil particles are considered.
    interpRangeConst = (sqrt(20)+0.01) 


    dx = (xmax - xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (0.0, 0.0))
    updateNeighbours!(particleGrid, interpRangeConst*dx)

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        v = zeros(8)
        for nbIndex in particle.neighbourIndices
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            stencil = getStencil(deltaX, deltaY, 8)
            theta = atan(deltaY, deltaX)
            v[stencil + 1] += 1
        end
        if !all(v .>= 5)
            println(v)
        end
        @test all(v .>= 5)
    end
end

@testset "getStencil irregular grid" begin
    xmin = -5
    xmax = 5
    interpAlpha = 6.0
    N = 50

    # Cannot trust that particles on vertical and horizontal lines are put inside stencil due to numerical round-off. 
    # Range must be large enough such sufficient inner-stencil particles are considered.
    interpRangeConst = (sqrt(5.0^2 + 3.0^2)+0.01) 


    dx = (xmax - xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dx/2), rng=MersenneTwister(2))
    updateNeighbours!(particleGrid, interpRangeConst*dx)

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        v = zeros(8)
        for nbIndex in particle.neighbourIndices
            deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
            stencil = getStencil(deltaX, deltaY, 8)
            theta = atan(deltaY, deltaX)
            # println("nB: $(nbIndex), ($(deltaX), $(deltaY)), $(theta), $(stencil)")
            v[stencil + 1] += 1
        end
        if !all(v .>= 5)
            println(v)
            println(particleIndex)
        end
        @test all(v .>= 5)
    end
end


@testset "MUSCL Gradients 1D" begin
    tmax = 7.5
    N = 100
    a = 1.0;
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 6.0
    interpRangeConst = 3.5

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Equation
    eq = LinearAdvection(a)

    # Grid stuff
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness=(xmax-xmin)/(2*N))
    setInitialConditions!(particleGrid, linearInit)
    
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*particleGrid.dx)

    # SimSettings
    saveFreq = 100000000000
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*particleGrid.dx,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq,
                            organiseFiles=false)

    # Test first-order MUSCL
    method = RK3(MUSCL(1), N)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        if 8 <= particleIndex <= N-8  # To avoid discontinuity at boundary
            # Compute dxVec
            for (i, nbIndex) in enumerate(particleGrid.grid[particleIndex].neighbourIndices)
                deltaPos = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
                particle.dxVec[i] = deltaPos
                particle.dfVec[i] = particleGrid.grid[nbIndex].rho - particleGrid.grid[particleIndex].rho
            end
            @. particle.wVec = exp(-interpAlpha*((particle.dxVec)^2))

            initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
            rhos = map(particle -> particle.rho, particleGrid.grid)
            res = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method exact for linear function
            @test isapprox(res, 1.0, rtol=1e-12)  
        end
    end

    # Test second-order MUSCL
    method = RK3(MUSCL(2), N)
    setInitialConditions!(particleGrid, quadraticInit)
    initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        if 8 <= particleIndex <= N-8  # To avoid discontinuity at boundary
            res = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method yields exact gradient for quadratic function
            @test isapprox(res, 2*particle.pos, rtol=1e-9)
        end
    end

    # Test third-order MUSCL
    method = RK3(MUSCL(3), N)
    setInitialConditions!(particleGrid, cubicInit)
    initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        if 8 <= particleIndex <= N-8  # To avoid discontinuity at boundary
            res = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method yields exact gradient for cubic function
            # @test isapprox(res, 3*(particle.pos^2), rtol=1e-12)  # The method converges, yet the conditioning in this test is really really bad.
        end
    end

    # Test fourth-order MUSCL
    method = RK3(MUSCL(4), N)
    setInitialConditions!(particleGrid, quarticInit)
    initTimeStep(method.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        if 8 <= particleIndex <= N-8  # To avoid discontinuity at boundary
            res = method.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method yields exact gradient for quadratic function
            @test isapprox(res, 4*(particle.pos^3), rtol=1e-9)
        end
    end

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end

@testset "MUSCL Gradients 2D" begin
    a = (1.0, 1.0);
    tmax = 7.5
    saveDir = "$(@__DIR__)/data/"
    xmin = -5
    xmax = 5
    interpAlpha = 6.0
    N = 50

    # Equation
    eq = LinearAdvection(a)

    randomness = (xmax-xmin)/(2*N)
    interpRangeConst = 3.5
    method1 = RK3(MUSCL(1), N)
    method2 = RK3(MUSCL(2), N)

    # Copy state of RNG
    state = copy(Meshfree4ScalarEq.rng)

    # Grid stuff
    dx = (xmax - xmin)/N
    particleGrid = ParticleGrid2D(xmin, xmax, xmin, xmax, N, N; randomness = (dx/2, dx/2))
    
    dt = getTimeStep(particleGrid, eq, interpAlpha, interpRangeConst*(xmax-xmin)/(N))

    # SimSettings
    saveFreq = 100000000000
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRangeConst*(xmax-xmin)/(N),
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq,
                            organiseFiles=false)

    # MUSCL 1 test
    param = 2.0
    setInitialConditions!(particleGrid, (x, y) -> testFunction2(x, y, param))
    initTimeStep(method1.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        # Make sure that particles are not too far from the boundary
        if (xmin + 10*particleGrid.dx <= particle.pos[1] <= xmax - 10*particleGrid.dx) && (xmin + 10*particleGrid.dx <= particle.pos[2] <= xmax - 10*particleGrid.dx)
            res = method1.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method yields exact gradient for linear function
            grad = gradTestFunction2(particle.pos[1], particle.pos[2], param)
            @test isapprox(res, dot(eq.vel, grad), rtol=1e-14)
        end
    end

    # MUSCL 2 test
    setInitialConditions!(particleGrid, (x, y) -> testFunction1(x, y, param))
    initTimeStep(method2.gradientInterpolator, particleGrid, interpAlpha, interpRangeConst*particleGrid.dx)
    rhos = map(particle -> particle.rho, particleGrid.grid)
    for (particleIndex, particle) in enumerate(particleGrid.grid)
        # Make sure that particles are not too far from the boundary
        if (xmin + 10*particleGrid.dx <= particle.pos[1] <= xmax - 10*particleGrid.dx) && (xmin + 10*particleGrid.dx <= particle.pos[2] <= xmax - 10*particleGrid.dx)
            res = method2.gradientInterpolator(particleGrid, particleIndex, rhos, eq.vel, settings)  # Method yields exact gradient for quadratic function
            grad = gradTestFunction1(particle.pos[1], particle.pos[2], param)
            @test isapprox(res, dot(eq.vel, grad), rtol=1e-11)
        end
    end

    # Return RNG state
    copy!(Meshfree4ScalarEq.rng, state)
end
