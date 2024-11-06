module ParticleGridStability

using LinearAlgebra
using SparseArrays
using ..Interpolations
using ..ParticleGrids

export computeODE

"""
    computeODE(particleGrid::ParticleGrid1D, vel::Real, alpha::Real, algorithm::String, order::Int64)

Compute linearised system for some spatial discretisations for the linear advection equations.
"""
function computeODE(particleGrid::ParticleGrid1D, vel::Real, alpha::Real, interpRange::Real, algorithm::String, order::Int64, weightFunction=exponentialWeightFunction())
    @assert algorithm in ["upwind", "central", "muscl"] "Algorithm not supported."

    # Some required variables
    I = Int64[]  # Row and column indices
    J = Int64[]
    V = Float64[]  # Values

    if algorithm == "central" || algorithm == "upwind"
        @assert (order == 1) || (order == 2) "Order must be one or two."
        for (particleIndex, particle) in enumerate(particleGrid.grid)
            
            # From correct set of neighbours, compute distances and weights
            if algorithm == "central"
                dxs = Vector{Float64}(undef, length(particle.neighbourIndices))
                ws = Vector{Float64}(undef, length(particle.neighbourIndices))
                acceptedNb = Vector{Int64}(undef, length(particle.neighbourIndices))
                for (i, nbIndex) in enumerate(particle.neighbourIndices)
                    dxs[i] = getPeriodicDistance(particleGrid, particleIndex, nbIndex)/particleGrid.dx
                    ws[i] = weightFunction(dxs[i]; param=alpha, normalisation=1.0)
                    acceptedNb[i] = nbIndex
                end
            elseif algorithm == "upwind"
                dxs = Vector{Float64}(undef, 0)
                ws = Vector{Float64}(undef, 0)
                acceptedNb = Vector{Int64}(undef, 0)
                for nbIndex in particle.neighbourIndices
                    dx = getPeriodicDistance(particleGrid, particleIndex, nbIndex)/particleGrid.dx
                    if ((dx > 0.0) && (vel < 0.0)) || ((dx < 0.0) && (vel > 0.0))
                        push!(dxs, dx)
                        push!(ws, weightFunction(dx; param=alpha, normalisation=1.0))
                        push!(acceptedNb, nbIndex)
                    end
                end
            end
            
            # Fill elements in ODE matrix
            if order == 1
                @assert length(dxs) >= 1 "Not enough neighbours found for particle $(particleIndex)."
                # Set diagonal elements
                denum = sum([ws[i]*(dxs[i]^2) for i in eachindex(dxs)])
                num = dot(dxs, ws)
                push!(J, particleIndex)
                push!(I, particleIndex)
                push!(V, vel*num/denum/particleGrid.dx)

                # Set off-diagonal elements
                for (i, nbIndex) in enumerate(acceptedNb)
                    push!(J, nbIndex)
                    push!(I, particleIndex)
                    push!(V, -vel*ws[i]*dxs[i]/denum/particleGrid.dx)
                end
            elseif order == 2
                @assert length(dxs) >= 2 "Not enough neighbours found for particle $(particleIndex)."
                A11 = sum([w*(dx^2) for (w, dx) in zip(ws, dxs)])
                A12 = sum([w*(dx^3) for (w, dx) in zip(ws, dxs)])/2
                A22 = sum([w*(dx^4) for (w, dx) in zip(ws, dxs)])/4
                D = A11*A22-(A12^2)

                # Set diagonal elements
                num = sum([A22*w*dx - 0.5*A12*w*(dx^2) for (w, dx) in zip(ws, dxs)])
                push!(J, particleIndex)
                push!(I, particleIndex)
                push!(V, vel*num/D/particleGrid.dx)

                # Set off-diagonal elements
                for (i, nbIndex) in enumerate(acceptedNb)
                    push!(J, nbIndex)
                    push!(I, particleIndex)
                    push!(V, -vel*(A22*ws[i]*dxs[i] - 0.5*A12*ws[i]*(dxs[i]^2))/D/particleGrid.dx)
                end
            end
        end
        A = sparse(I, J, V)
        @assert all(.!isinf.(A)) "There is an inf."
        @assert all(.!isnan.(A)) "There is a Nan."
        @assert all(isapprox.(sum(A, dims=2), 0.0, atol=1e-12)) "$(maximum(abs.(sum(A, dims=2))))"
        return A
    elseif algorithm == "muscl"
        @assert (order == 1) || (order == 2) || (order == 3) || (order == 4) "Order must be one, two, three or four."
        A = sparse(zeros(length(particleGrid.grid), length(particleGrid.grid)))

        if order == 1
            muscl = MUSCL(1; weightFunction=weightFunction)
            initTimeStep(muscl, particleGrid, alpha, interpRange)  # populate particle.alfaij vectors

            for (particleIndex, particle) in enumerate(particleGrid.grid)
                for (jIndex, jNbIndex) in enumerate(particle.neighbourIndices)
                    deltaPos = getPeriodicDistance(particleGrid, particleIndex, jNbIndex)
                    jNbParticle = particleGrid.grid[jNbIndex]
    
                    if vel*deltaPos > 0.0
                        for (kIndex, kNbIndex) in enumerate(particle.neighbourIndices)
                            A[particleIndex, kNbIndex] += 0.5*particle.alfaij[jIndex]*vel*deltaPos*particle.alfaij[kIndex]
                        end
                        A[particleIndex, particleIndex] += -0.5*particle.alfaij[jIndex]*vel*deltaPos*sum(particle.alfaij)
                    else
                        A[particleIndex, particleIndex] += -vel*particle.alfaij[jIndex]
                        for (kIndex, kNbIndex) in enumerate(jNbParticle.neighbourIndices)
                            A[particleIndex, kNbIndex] += -0.5*vel*particle.alfaij[jIndex]*deltaPos*jNbParticle.alfaij[kIndex]
                        end
                        A[particleIndex, jNbIndex] += vel*particle.alfaij[jIndex]*(1.0 + 0.5*deltaPos*sum(jNbParticle.alfaij))
                    end
                end
            end    
        elseif order == 2
            muscl = MUSCL(2; weightFunction=weightFunction)
            initTimeStep(muscl, particleGrid, alpha, interpRange)  

            for (particleIndex, particle) in enumerate(particleGrid.grid)
                for (jIndex, jNbIndex) in enumerate(particle.neighbourIndices)
                    deltaPos = getPeriodicDistance(particleGrid, particleIndex, jNbIndex)
                    jNbParticle = particleGrid.grid[jNbIndex]
    
                    # Quadratic reconstruction
                    if vel*deltaPos > 0.0
                        for (kIndex, kNbIndex) in enumerate(particle.neighbourIndices)
                            A[particleIndex, kNbIndex] += particle.alfaijBar[jIndex]*vel*(deltaPos*particle.alfaijBar[kIndex]/2 + (deltaPos^2)*particle.betaij[kIndex]/8)
                        end
                        A[particleIndex, particleIndex] += -particle.alfaijBar[jIndex]*vel*(deltaPos*sum(particle.alfaijBar)/2 + (deltaPos^2)*sum(particle.betaij)/8)
                    else
                        A[particleIndex, particleIndex] += -vel*particle.alfaijBar[jIndex]
                        for (kIndex, kNbIndex) in enumerate(jNbParticle.neighbourIndices)
                            A[particleIndex, kNbIndex] += vel*particle.alfaijBar[jIndex]*(-0.5*deltaPos*jNbParticle.alfaijBar[kIndex] + jNbParticle.betaij[kIndex]*(deltaPos^2)/8)
                        end
                        A[particleIndex, jNbIndex] += vel*particle.alfaijBar[jIndex]*(1.0 + 0.5*deltaPos*sum(jNbParticle.alfaijBar) - (deltaPos^2)*sum(jNbParticle.betaij)/8)
                    end

                end
            end    
        elseif order == 3
            muscl = MUSCL(3; weightFunction=weightFunction)
            initTimeStep(muscl, particleGrid, alpha, interpRange)  

            for (particleIndex, particle) in enumerate(particleGrid.grid)
                for (jIndex, jNbIndex) in enumerate(particle.neighbourIndices)
                    deltaPos = getPeriodicDistance(particleGrid, particleIndex, jNbIndex)
                    jNbParticle = particleGrid.grid[jNbIndex]
    
                    # Cubic reconstruction
                    if vel*deltaPos > 0.0
                        for (kIndex, kNbIndex) in enumerate(particle.neighbourIndices)
                            A[particleIndex, kNbIndex] += particle.alfaijBar[jIndex]*vel*(deltaPos*particle.alfaijBar[kIndex]/2 + (deltaPos^2)*particle.betaij[kIndex]/8 + (deltaPos^3)*particle.alfaij[kIndex]/(8*6))
                        end
                        A[particleIndex, particleIndex] += -particle.alfaijBar[jIndex]*vel*(deltaPos*sum(particle.alfaijBar)/2 + (deltaPos^2)*sum(particle.betaij)/8 + (deltaPos^3)*sum(particle.alfaij)/(8*6))
                    else
                        A[particleIndex, particleIndex] += -vel*particle.alfaijBar[jIndex]
                        for (kIndex, kNbIndex) in enumerate(jNbParticle.neighbourIndices)
                            A[particleIndex, kNbIndex] += vel*particle.alfaijBar[jIndex]*(-0.5*deltaPos*jNbParticle.alfaijBar[kIndex] + jNbParticle.betaij[kIndex]*(deltaPos^2)/8 - jNbParticle.alfaij[kIndex]*(deltaPos^3)/(8*6))
                        end
                        A[particleIndex, jNbIndex] += vel*particle.alfaijBar[jIndex]*(1.0 + 0.5*deltaPos*sum(jNbParticle.alfaijBar) - (deltaPos^2)*sum(jNbParticle.betaij)/8 + (deltaPos^3)*sum(jNbParticle.alfaij)/(8*6))
                    end
                end
            end    
        elseif order == 4
            muscl = MUSCL(4; weightFunction=weightFunction)
            initTimeStep(muscl, particleGrid, alpha, interpRange)  

            for (particleIndex, particle) in enumerate(particleGrid.grid)
                for (jIndex, jNbIndex) in enumerate(particle.neighbourIndices)
                    deltaPos = getPeriodicDistance(particleGrid, particleIndex, jNbIndex)
                    jNbParticle = particleGrid.grid[jNbIndex]
    
                    # Quartic reconstruction
                    if vel*deltaPos > 0.0
                        for (kIndex, kNbIndex) in enumerate(particle.neighbourIndices)
                            A[particleIndex, kNbIndex] += particle.alfaijBar[jIndex]*vel*(deltaPos*particle.alfaijBar[kIndex]/2 + (deltaPos^2)*particle.betaij[kIndex]/8 + (deltaPos^3)*particle.alfaij[kIndex]/(8*6) + (deltaPos^4)*particle.gammaij[kIndex]/((2^4)*24))
                        end
                        A[particleIndex, particleIndex] += -particle.alfaijBar[jIndex]*vel*(deltaPos*sum(particle.alfaijBar)/2 + (deltaPos^2)*sum(particle.betaij)/8 + (deltaPos^3)*sum(particle.alfaij)/(8*6) + (deltaPos^4)*sum(particle.gammaij)/((2^4)*24))
                    else
                        A[particleIndex, particleIndex] += -vel*particle.alfaijBar[jIndex]
                        for (kIndex, kNbIndex) in enumerate(jNbParticle.neighbourIndices)
                            A[particleIndex, kNbIndex] += vel*particle.alfaijBar[jIndex]*(-0.5*deltaPos*jNbParticle.alfaijBar[kIndex] + jNbParticle.betaij[kIndex]*(deltaPos^2)/8 - jNbParticle.alfaij[kIndex]*(deltaPos^3)/(8*6) + jNbParticle.gammaij[kIndex]*(deltaPos^4)/((2^4)*24))
                        end
                        A[particleIndex, jNbIndex] += vel*particle.alfaijBar[jIndex]*(1.0 + 0.5*deltaPos*sum(jNbParticle.alfaijBar) - (deltaPos^2)*sum(jNbParticle.betaij)/8 + (deltaPos^3)*sum(jNbParticle.alfaij)/(8*6) - (deltaPos^4)*sum(jNbParticle.gammaij)/((2^4)*24))
                    end
                end
            end    
        end

        @assert all(.!isinf.(A)) "There is an inf."
        @assert all(.!isnan.(A)) "There is a Nan."
        # @assert all(isapprox.(sum(A, dims=2), 0.0, atol=1e-12)) "$(maximum(sum(A, dims=2)))"
        return -A*2  # Minus sign normally taken into account in the time stepper routine. x2 because the midpoints are used to compute the gradient and not the actual points (stencils scale by 0.5).
    end
end

function computeODE(particleGrid::ParticleGrid2D, vel::Tuple{Real, Real}, alpha::Real, interpRange::Real, algorithm::String, order::Int64, weightFunction=exponentialWeightFunction)
    @assert algorithm in ["upwindClassic", "upwindPraveen", "muscl", "AxelMUSCL"] "Algorithm not supported."
    @assert (order == 1) || (order == 2) "Order must be one or two."

    # Some required variables
    I = Int64[]  # Row and column indices
    J = Int64[]
    V = Float64[]  # Values

    if algorithm == "upwindClassic"
        for (particleIndex, particle) in enumerate(particleGrid.grid)
            dxVec = Vector{Float64}(undef, 0)
            dyVec = Vector{Float64}(undef, 0)
            nbIndices = Vector{Int64}(undef, 0)

            # Select correclty oriented neighbours
            for nbIndex in particleGrid.grid[particleIndex].neighbourIndices
                deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
                if deltaX*vel[1] + deltaY*vel[2] < 0
                    push!(dxVec, deltaX/interpRange)
                    push!(dyVec, deltaY/interpRange)
                    push!(nbIndices, nbIndex)
                end
            end
            wVec = weightFunction(dxVec, dyVec; param=alpha, normalisation=1.0)

            # Compute coefficients
            if order == 1
                A = Matrix{Float64}(undef, length(dxVec), 2)
                @. A[:, 1] = dxVec
                @. A[:, 2] = dyVec
                coeff = pinv(A)
                coeff[1, :] .= coeff[1, :] .* wVec
                coeff[2, :] .= coeff[2, :] .* wVec
            elseif order == 2
                A = Matrix{Float64}(undef, length(dxVec), 5)
                @. A[:, 1] = dxVec * wVec
                @. A[:, 2] = dyVec * wVec
                @. A[:, 3] = (dxVec^2) * wVec / 2
                @. A[:, 4] = (dyVec^2) * wVec / 2
                @. A[:, 5] = dxVec * dyVec * wVec
                coeff = pinv(A)
                coeff[1, :] .= coeff[1, :] .* wVec
                coeff[2, :] .= coeff[2, :] .* wVec
            end

            # Fill in elements in matrix
            for (k, nbIndex) in enumerate(nbIndices)
                push!(I, particleIndex)
                push!(J, nbIndex)
                push!(V, - vel[1]*coeff[1, k]/interpRange - vel[2]*coeff[2, k]/interpRange)
            end

            # Fill in diagonal
            push!(I, particleIndex)
            push!(J, particleIndex)
            push!(V, vel[1]*sum(coeff[1, :])/interpRange + vel[2]*sum(coeff[2, :])/interpRange)
            
        end
        return sparse(I, J, V)
    elseif algorithm == "upwindPraveen"
        @assert order == 1
        for (particleIndex, particle) in enumerate(particleGrid.grid)
            # Create 2x2 LS system
            A11 = A12 = A22 = 0.0
            for nbIndex in particle.neighbourIndices
                deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
                w = weightFunction(deltaX, deltaY; param=alpha, normalisation=interpRange)
                A11 += w*(deltaX^2) 
                A12 += w*deltaX*deltaY
                A22 += w*(deltaY^2)
            end
            D = A11*A22 - (A12^2)
            
            sum = 0.0
            for nbIndex in particle.neighbourIndices
    
                # Solve 2x2 LS system
                deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, nbIndex)
                w = weightFunction(deltaX, deltaY; param=alpha, normalisation=interpRange)
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
                sum += 2*cij
                push!(I, particleIndex)
                push!(J, nbIndex)
                push!(V, -2*cij)
            end
            push!(I, particleIndex)
            push!(J, particleIndex)
            push!(V, sum)
        end
        return sparse(I, J, V)
    elseif algorithm == "muscl"
        muscl = MUSCL(order; weightFunction=weightFunction)
        initTimeStep(muscl, particleGrid, alpha, interpRange)  # populate particle.alfaij vectors
        for (particleIndex, particle) in enumerate(particleGrid.grid)
            # Store values at row particleIndex in dict
            d = Dict{Int64, Float64}()
            d[particleIndex] = 0.0
            for jNbIndex in particle.neighbourIndices
                d[jNbIndex] = 0.0
                for kNbIndex in particleGrid.grid[jNbIndex].neighbourIndices
                    d[kNbIndex] = 0.0
                end
            end

            # Add contributions to row
            for (j, jNbIndex) in enumerate(particle.neighbourIndices)
                deltaX, deltaY = getPeriodicDistance(particleGrid, particleIndex, jNbIndex)
                if deltaX*vel[1] + deltaY*vel[2] > 0.0
                    s = 0.0
                    for (k, kNbIndex) in enumerate(particle.neighbourIndices)
                        if order == 1
                            t = -vel[1]*particle.alfaij[j]*(deltaX*particle.alfaij[k] + deltaY*particle.betaij[k]) - vel[2]*particle.betaij[j]*(deltaX*particle.alfaij[k] + deltaY*particle.betaij[k])
                        elseif order == 2
                            t = -vel[1]*particle.alfaij[j]*(deltaX*particle.alfaij[k] + deltaY*particle.betaij[k] + 0.25*(deltaX^2)*particle.alfaijBar[k] + 0.25*(deltaY^2)*particle.betaijBar[k] + 0.5*deltaX*deltaY*particle.gammaij[k]) 
                            t += -vel[2]*particle.betaij[j]*(deltaX*particle.alfaij[k] + deltaY*particle.betaij[k] + 0.25*(deltaX^2)*particle.alfaijBar[k] + 0.25*(deltaY^2)*particle.betaijBar[k] + 0.5*deltaX*deltaY*particle.gammaij[k])
                        end
                        d[kNbIndex] += t
                        s += t
                    end
                    d[particleIndex] += -s
                else
                    jParticle = particleGrid.grid[jNbIndex]
                    d[particleIndex] += 2*vel[1]*particle.alfaij[j]+2*vel[2]*particle.betaij[j]

                    if order == 1
                        d[jNbIndex] += (-1.0 - 0.5*deltaX*sum(jParticle.alfaij) - 0.5*deltaY*sum(jParticle.betaij))*(particle.alfaij[j]*vel[1] + particle.betaij[j]*vel[2])*2
                        for (k, kNbIndex) in enumerate(jParticle.neighbourIndices)
                            d[kNbIndex] += (vel[1]*particle.alfaij[j] + vel[2]*particle.betaij[j])*(deltaX*jParticle.alfaij[k] + deltaY*jParticle.betaij[k])
                        end
                    elseif order == 2
                        d[jNbIndex] += (-1.0 - 0.5*deltaX*sum(jParticle.alfaij) - 0.5*deltaY*sum(jParticle.betaij) + (deltaX^2)*sum(jParticle.alfaijBar)/8 + (deltaY^2)*sum(jParticle.betaijBar)/8 + deltaX*deltaY*sum(jParticle.gammaij)/4)*(particle.alfaij[j]*vel[1] + particle.betaij[j]*vel[2])*2
                        for (k, kNbIndex) in enumerate(jParticle.neighbourIndices)
                            d[kNbIndex] += (vel[1]*particle.alfaij[j] + vel[2]*particle.betaij[j])*(deltaX*jParticle.alfaij[k] + deltaY*jParticle.betaij[k] - (deltaX^2)*jParticle.alfaijBar[k]/4 - (deltaY^2)*jParticle.betaijBar[k]/4 - deltaX*deltaY*jParticle.gammaij[k]/2)
                        end
                    end
                end
            end

            # Add dict entries to I, J and V arrays
            for (key, value) in d
                if value != 0.0
                    push!(I, particleIndex)
                    push!(J, key)
                    push!(V, value)
                end
            end
        end
        return sparse(I, J, V)
    elseif algorithm == "AxelMUSCL"
        muscl = AxelMUSCL(1; weightFunction=weightFunction)
        initTimeStep(muscl, particleGrid, alpha, interpRange)  # populate particle.alfaij vectors
        for (particleIndex, particle) in enumerate(particleGrid.grid)
            # Store values at row particleIndex in dict
            d = Dict{Int64, Float64}()
            d[particleIndex] = 0.0
            for jNbIndex in particle.neighbourIndices
                d[jNbIndex] = 0.0
                for kNbIndex in particleGrid.grid[jNbIndex].neighbourIndices
                    d[kNbIndex] = 0.0
                end
            end

            alfai = particle.betaijBar[1]
            for (index, nbIndex) in enumerate(particle.neighbourIndices)
                nx = particle.dxVec[index]/sqrt(particle.dxVec[index]^2 + particle.dyVec[index]^2)
                ny = particle.dyVec[index]/sqrt(particle.dxVec[index]^2 + particle.dyVec[index]^2)
                
                # Linear reconstruction from particleIndex and neighbour at center point 
                inn = (vel[1]*nx + vel[2]*ny)/(2*pi)
                if inn > 0.0
                    # contribution to particleIndex
                    d[particleIndex] += inn*particle.gammaij[index]*(1 - alfai)*sum((nx*particle.alfaij[i] + ny*particle.betaij[i]) for (i, _) in enumerate(particle.neighbourIndices))

                    # Contribution to neighbours
                    for (jIndex, jNbIndex) in enumerate(particle.neighbourIndices)
                        d[jNbIndex] += inn*particle.gammaij[index]*alfai*(nx*particle.alfaij[jIndex] + ny*particle.betaij[jIndex])
                    end
                    # fij = fVec[particleIndex] + alfai*sum((nx*particle.alfaij[i] + ny*particle.betaij[i])*(fVec[k] - fVec[particleIndex]) for (i, k) in enumerate(particle.neighbourIndices))
                else
                    nbParticle = particleGrid.grid[nbIndex]
                    d[nbIndex] += inn*particle.gammaij[index]*(1 - sum(((nx*alfai - particle.dxVec[index])*nbParticle.alfaij[i] + (ny*alfai - particle.dyVec[index])*nbParticle.betaij[i]) for (i, _) in enumerate(nbParticle.neighbourIndices)))
                    for (jIndex, jNbIndex) in enumerate(nbParticle.neighbourIndices)
                        d[jNbIndex] += inn*particle.gammaij[index]*alfai*((nx*alfai - particle.dxVec[index])*nbParticle.alfaij[jIndex] + (ny*alfai - particle.dyVec[index])*nbParticle.betaij[jIndex])
                    end
                    # fij = fVec[nbIndex] + sum(((nx*alfai - particle.dxVec[index])*nbParticle.alfaij[i] + (ny*alfai - particle.dyVec[index])*nbParticle.betaij[i])*(fVec[k] - fVec[nbIndex]) for (i, k) in enumerate(nbParticle.neighbourIndices))
                end
            end
            # Add dict entries to I, J and V arrays
            for (key, value) in d
                if value != 0.0
                    push!(I, particleIndex)
                    push!(J, key)
                    push!(V, value)
                end
            end
        end

        return sparse(I, J, V)
    end
end


end  # module