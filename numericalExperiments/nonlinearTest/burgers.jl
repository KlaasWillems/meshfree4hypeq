using Meshfree4ScalarEq.ScalarHyperbolicEquations
using Meshfree4ScalarEq.ParticleGrids
using Meshfree4ScalarEq.TimeIntegration
using Meshfree4ScalarEq.Interpolations
using Meshfree4ScalarEq.SimSettings
using Meshfree4ScalarEq

function smoothInit1(x::Real)
    return exp(-x^2)
end

function smoothInit2(x::Real)
    return sin(2*π*x/5) + 1.0
end

function shockInit(x::Real)
    return x > 0.0 ? 1.0 : 0.0
end

function doSimluation(method::String, initFunc::String, randomnessFactor::Real = 1/4)
    # Simulation settings
    CFL = 1/5
    tmax = 10.0
    N = 100
    xmin = -5
    xmax = 5
    saveDir = "$(@__DIR__)/data/"
    interpAlpha = 1.0
    saveFreq = 5

    state = copy(Meshfree4ScalarEq.rng)

    # Simlation algorithms
    if method == "muscl2"
        method = RalstonRK2(MUSCL(2), N; mood=MOODu1(;deltaRelax=true))
        saveDir *= "RK2MUSCL2"
    elseif method == "lf"
        method = LaxFriedrich(N)
        saveDir *= "LF"
    end

    # Equation
    eq = BurgersEquation()
    eqLin = LinearAdvection(1.0)
    
    # Grid stuff
    dx = (xmax-xmin)/N
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = randomnessFactor*dx)
    saveDir *= "_irgrid"

    interpRange = 3.5*particleGrid.dx
    dt = CFL*getTimeStep(particleGrid, eqLin, interpAlpha, interpRange)  # Time step such that meshfree first order least squares method is positive. (Check MeshfreeUpwind_irgrid_shockInit)

    # Initial condition
    if initFunc == "smoothInit1"
        setInitialConditions!(particleGrid, smoothInit1)
        saveDir *= "_smoothInit1"
    elseif initFunc == "smoothInit2"
        setInitialConditions!(particleGrid, smoothInit2)
        saveDir *= "_smoothInit2"
    elseif initFunc == "shockInit"
        setInitialConditions!(particleGrid, shockInit)
        saveDir *= "_shockInit"
    end

    # Create SimSettings object
    settings = SimSetting(  tmax=tmax,
                            dt=dt,
                            interpRange=interpRange,
                            interpAlpha=interpAlpha,
                            saveDir=saveDir*"/", 
                            saveFreq=saveFreq)

    # Do simulation
    mainTimeIntegrator!(method, eq, particleGrid, settings)

    # Generate plots
    for i = 1:10
        if i < settings.currentSaveNb-1
            plotDensity(settings, i; saveFigure=true, showMOODEvents=true)
        end
    end
    plotDensity(settings, settings.currentSaveNb-1; saveFigure=true, showMOODEvents=true)
    animateDensity(settings; saveFigure=true, fps=2, showMOODEvents=true)

    copy!(Meshfree4ScalarEq.rng, state)
end

doSimluation("muscl2", "shockInit")
doSimluation("muscl2", "smoothInit1")
doSimluation("muscl2", "smoothInit2")
doSimluation("lf", "shockInit", 0)
doSimluation("lf", "smoothInit1", 0)
doSimluation("lf", "smoothInit2", 0)
