using Statistics
using Test
using Meshfree4ScalarEq.Particles
using Meshfree4ScalarEq.ParticleGrids

@testset "Update Neighbours" begin
    xmin = -1.0
    xmax = 2.0
    N = 10
    particleGrid = ParticleGrid1D(xmin, xmax, N; randomness = 0.0)
    dx = particleGrid.dx
    @test issorted(particleGrid.grid, by=particle->particle.pos) 
    
    # Check neighbours
    updateNeighbours!(particleGrid, dx*2.5)

    for particleIndex in eachindex(particleGrid.grid)
        particle = particleGrid.grid[particleIndex]

        @test length(particle.neighbourIndices) == 4

        # Check neighbours indices
        if particleIndex == 1
            @test 2 in particle.neighbourIndices
            @test 3 in particle.neighbourIndices
            @test particleGrid.N in particle.neighbourIndices
            @test particleGrid.N-1 in particle.neighbourIndices
        elseif particleIndex == 2
            @test 3 in particle.neighbourIndices
            @test 4 in particle.neighbourIndices
            @test particleGrid.N in particle.neighbourIndices
            @test 1 in particle.neighbourIndices
        elseif particleIndex == particleGrid.N
            @test 1 in particle.neighbourIndices
            @test 2 in particle.neighbourIndices
            @test particleGrid.N-1 in particle.neighbourIndices
            @test particleGrid.N-2 in particle.neighbourIndices
        elseif particleIndex == particleGrid.N-1
            @test 1 in particle.neighbourIndices
            @test particleGrid.N in particle.neighbourIndices
            @test particleGrid.N-3 in particle.neighbourIndices
            @test particleGrid.N-2 in particle.neighbourIndices
        else
            @test particleIndex+1 in particle.neighbourIndices
            @test particleIndex+2 in particle.neighbourIndices
            @test particleIndex-1 in particle.neighbourIndices
            @test particleIndex-2 in particle.neighbourIndices
        end
    end
end

@testset "getPeriodicDistance" begin
    xmin = -1.0
    xmax = 5.0
    regular = false

    p1 = Particle1D(xmin, 0.0, true)
    p2 = Particle1D(-0.2, 0.0, false)
    p3 = Particle1D(0.5, 0.0, false)
    p4 = Particle1D(2.5, 0.0, false)
    p5 = Particle1D(3.6, 0.0, false)
    p6 = Particle1D(4.9, 0.0, false)
    grid = [p1; p2; p3; p4; p5; p6]
    dx = mean(diff(map(particle -> particle.pos, grid)))

    particleGrid = ParticleGrid1D(grid, xmin, xmax, dx, regular)

    # Some handchecked tests
    @test isapprox(getPeriodicDistance(particleGrid, 1, 2), 0.8)
    @test isapprox(getPeriodicDistance(particleGrid, 1, 3), 1.5)
    @test isapprox(getPeriodicDistance(particleGrid, 1, 5), -1.4)
    @test isapprox(getPeriodicDistance(particleGrid, 1, 6), -0.1)
    @test isapprox(getPeriodicDistance(particleGrid, 1, 4), -2.5)
    @test isapprox(getPeriodicDistance(particleGrid, 2, 5), -2.2)
    @test isapprox(getPeriodicDistance(particleGrid, 3, 5), -2.9)
    @test isapprox(getPeriodicDistance(particleGrid, 4, 5), 1.1)

    for i in eachindex(grid)
        for j in eachindex(grid)
            @test isapprox(getPeriodicDistance(particleGrid, i, j), -getPeriodicDistance(particleGrid, j, i))
        end
    end

end

@testset "updateVoxelInformation" begin
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 5
    Nx = 10
    Ny = 10
    particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny)
    maxDist = 1.0
    updateVoxelInformation!(particleGrid, maxDist)
    nbBoxesX = ceil((particleGrid.xmax - particleGrid.xmin)/maxDist)

    # Some manual tests
    for particle in particleGrid.grid
        for j = 1:5
            for i = 1:10
                if ((i-1) <= particle.pos[1] < i) && ( j-1 <= particle.pos[2] < j)
                    @test particle.voxel == (i-1) + (j-1)*nbBoxesX # "$(particle.pos), $(particle.voxel), $((i-1) + (j-1)*nbBoxesX)"
                end
            end
        end
    end
end

@testset "Grid indexing" begin
    Nx = 10
    Ny = 7
    nbBoxX = 10

    # Convert coordinates to linear index and then back to grid index.
    for i = 0:Nx-1
        for j = 0:Ny-1
            linear = gridToLinearIndex(i, j, nbBoxX)
            it, jt = linearIndexToGrid(linear, nbBoxX)
            @test (it == i) && (jt == j)
        end
    end
end

@testset "findNeighbouringVoxels" begin
    Nx = 4
    Ny = 5

    tup = findNeighbouringVoxels(19, Nx, Ny)
    @test 2 in tup
    @test 3 in tup
    @test 0 in tup
    @test 16 in tup
    @test 12 in tup
    @test 18 in tup
    @test 14 in tup
    @test 15 in tup

    tup = findNeighbouringVoxels(3, Nx, Ny)
    @test 2 in tup
    @test 6 in tup
    @test 7 in tup
    @test 0 in tup
    @test 4 in tup
    @test 18 in tup
    @test 19 in tup
    @test 16 in tup
end

@testset "updateNeighbours!" begin
    xmin = 0
    xmax = 10
    ymin = 0
    ymax = 5
    Nx = 10
    Ny = 10
    particleGrid = ParticleGrid2D(xmin, xmax, ymin, ymax, Nx, Ny)
    maxDist = 1.5
    updateNeighbours!(particleGrid, maxDist)

    for (particleIndex, particle) in enumerate(particleGrid.grid)
        for (nbParticleIndex, nbParticle) in enumerate(particleGrid.grid)
            d = getEuclideanDistance(particleGrid, particleIndex, nbParticleIndex)
            if (d <= maxDist) && (particleIndex != nbParticleIndex)
                @test nbParticleIndex in particle.neighbourIndices
            else
                @test !(nbParticleIndex in particle.neighbourIndices)
            end
        end
    end
end