abstract type Network end

function rand(m::Network) end

function sample!(m::Network) end


"""
    A network with i.i.d Bernoulli connections: A[n, m] ~ Bernoulli(ρ).
"""
struct BernoulliNetwork
    # parameters
    ρ
    N
    # hyperparameters
    α
    β
end

function rand(network::BernoulliNetwork)
    ρ = rand(Beta(network.α, network.β))
    return rand(Bernoulli.(ρ), network.N, network.N)
end

function rand(network::BernoulliNetwork, n::Int32)
    data = zeros(network.N, network.N, n)
    for i = 1:n
        data[:, :, i] = rand(network)
    end
    return data
end

likelihood(network::BernoulliNetwork, A, ρ) = ρ * ones(network.N, network.N)
likelihood(network::BernoulliNetwork, a, index) = network.ρ
function sample!(network::BernoulliNetwork, A)
    N = network.N
    network.ρ = rand(Beta(network.α + sum(A), network.β + sum(A) - N * N))
    return copy(network.ρ)
end


"""
    A network with conditionally i.i.d Bernoulli connections.

Each node is associated with a latent "block". Connections from block `i` to block `j` are i.i.d Bernoulli distributed. Specifically,

    (model here...)
"""
struct StochasticBlockNetwork
    # parameters
    ρ
    π
    z
    N
    # hyperparameters
    α
    β
    γ
end

function rand(network::StochasticBlockNetwork)
    K = length(network.γ)
    π = rand(Dirichlet(network.γ))
    z = dense(rand(Discrete(network.π), network.N))
    ρ = rand(Beta(network.α, network.β), K, K)
    A = zeros(network.N, network.N)
    for i = 1:network.N
        for j = 1:network.N
            zi = network.z[i]
            zj = network.z[j]
            A[i, j] = rand.(Bernoulli.(ρ[zi, zj]))
        end
    end
    return A
end

function rand(network::StochasticBlockNetwork, n::Int32)
    data = zeros(network.N, network.N, n)
    for i = 1:n
        data[:, :, i] = rand(network)
    end
    return data
end


function likelihood(network::StochasticBlockNetwork, A)
    p = 1.
    for i = 1:network.N
        for j = 1:network.N
            ki = network.z[i]
            kj = network.z[j]
            p *= ρ[ki, kj]
        end
    end
    return p
end

function likelihood(network::StochasticBlockNetwork, a, index)
    ki, kj = network.z[collect(index)]
    return ρ[CartesianIndex(tuple(ki, kj)...)]
end

function sample_ρ!(network::StochasticBlockNetwork, A)
    K = length(network.γ)
    counts = zeros(K, K)
    connections = zeros(K, K)
    for i = 1:N
        for j = 1:N
            ki = network.z[i]
            kj = network.z[j]
            counts[ki, kj] += 1
            connections[ki, kj] += A[i, j]
        end
    end
    network.ρ = rand(Beta.(network.α .+ connections, network.β .+ counts .- connections))
    return copy(network.ρ)
end

function sample_z!(network::StochasticBlockNetwork)
    products = ones(length(p.π), network.N)
    for i = 1:network.N
        for j = 1:network.N
            kn = z[j]
            products[:, i] .*= ρ[kn, :] .* ρ[:, kn]
        end
    end
    network.z = rand(Discrete.(network.π .* products))
    return copy(network.z)
end

function sample_π!(network::StochasticBlockNetwork)
    network.π = rand(Dirichlet(network.γ + rangecount(z, 1:length(network.γ))))
    return copy(network.π)
end

function rangecount(x, range)
    counts = zeros(length(range))
    for value in x
        if value in range
            counts[value] += 1
        end
    end
    return counts
end

function sample!(network::StochasticBlockNetwork, A)
    ρ = sample_ρ!(network, A)
    z = sample_z!(network)
    π = sample_π!(network)
    return z, π, ρ
end
