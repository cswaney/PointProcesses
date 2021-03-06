"""Methods to implement Gibbs algorithm for network models."""


"""
    sample_connection_probability!(net::DenseNetwork, data)

Return a sample from the conditional posterior of `ρ`.

# Arguments
- `data::Array{Bool,2}`: `N x N` matrix of observed links.
"""
sample_network!(net::DenseNetwork, data) = ones(net.N, net.N)
sample_network(net::DenseNetwork, data) = ones(net.N, net.N)


"""
    sample_connection_probability!(net::BernoulliNetwork, data)

Return a sample from the conditional posterior of `ρ`.

# Arguments
- `data::Array{Bool,2}`: `N x N` matrix of observed links.
"""
function sample_network!(net::BernoulliNetwork, data)
    net.ρ = rand(Beta(net.α + sum(data), net.β + net.N * net.N - sum(data)))
    return copy(net.ρ)
end

function sample_network(net::BernoulliNetwork, data)
    return rand(Beta(net.α + sum(data), net.β + net.N * net.N - sum(data)))
end


"""
    sample_connection_probability!(net::StochasticBlockNetwork, data)

Return a sample from the conditional posterior of `ρ`.

# Arguments
- `data::Array{Bool,2}`: `N x N` matrix of observed links.
"""
function sample_network!(net::StochasticBlockNetwork, data)
    ρ = sample_connection_probability!(net, data)
    z = sample_classes!(net)
    π = sample_class_probability!(net)
    return ρ, z, π
end

function sample_network(net::StochasticBlockNetwork, data)
    return rand(Beta(net.α + sum(data), net.β + net.N * net.N - sum(data)))
end


"""
    sample_connection_probability!(net::StochasticBlockNetwork, data)

Return a sample from the conditional posterior of `ρ`.

# Arguments
- `data::Array{Bool,2}`: `N x N` matrix of observed links.
"""
function sample_connection_probability!(net::StochasticBlockNetwork, data)
    counts = zeros(net.K, net.K)
    connections = zeros(net.K, net.K)
    for i = 1:net.N
        for j = 1:net.N
            ki = net.z[i]
            kj = net.z[j]
            counts[ki, kj] += 1
            connections[ki, kj] += data[i, j]
        end
    end
    net.ρ = rand.(Beta.(net.α .+ connections, net.β .+ counts .- connections))
    return copy(net.ρ)
end

function sample_connection_probability(net::StochasticBlockNetwork, data)
    counts = zeros(net.K, net.K)
    connections = zeros(net.K, net.K)
    for i = 1:net.N
        for j = 1:net.N
            ki = net.z[i]
            kj = net.z[j]
            counts[ki, kj] += 1
            connections[ki, kj] += data[i, j]
        end
    end
    return rand.(Beta.(net.α .+ connections, net.β .+ counts .- connections))
end


"""
    sample_classes!(net::StochasticBlockNetwork)

Return a sample from the conditional posterior distribution of latent classes."""
function sample_classes!(net::StochasticBlockNetwork)
    products = [ones(net.K) for _ in 1:net.N]
    for i = 1:net.N
        for j = 1:net.N
            kn = net.z[j]
            products[i] .*= net.ρ[kn, :] .* net.ρ[:, kn]
        end
    end
    π_raw = [net.π .* prds for prds in products]
    z_raw = [sum(pr) for pr in π_raw]
    π = π_raw ./ z_raw
    net.z = dense(hcat(rand.(Discrete.(π))...))
    return copy(net.z)
end

function sample_classes(net::StochasticBlockNetwork)
    products = [ones(net.K) for _ in 1:net.N]
    for i = 1:net.N
        for j = 1:net.N
            kn = net.z[j]
            products[i] .*= net.ρ[kn, :] .* net.ρ[:, kn]
        end
    end
    π_raw = [net.π .* prds for prds in products]
    z_raw = [sum(pr) for pr in π_raw]
    π = π_raw ./ z_raw
    return dense(hcat(rand.(Discrete.(π))...))
end


"""
    sample_class_probability!(net::StochasticBlockNetwork)

Return a sample from the conditional posterior distribution of class probability.
"""
function sample_class_probability!(net::StochasticBlockNetwork)
    net.π = rand(Dirichlet(net.γ .+ rangecount(net.z, 1:net.K)))
    return copy(net.π)
end

function sample_class_probability(net::StochasticBlockNetwork)
    return rand(Dirichlet(net.γ .+ rangecount(net.z, 1:net.K)))
end


"""
    rangecount(x::Array{Int64,1}, r::UnitRange{Int64})

Return the number of elements in `x` equal to each value in range `r`, which is assumed to start from 1.

# Example
```
x = [1, 3, 2, 2, 1, 4]
r = 1:4
rangecount(x, r)  # [2, 2, 1, 1]
```
"""
function rangecount(x::Array{Int64,1}, r::UnitRange{Int64})
    r[1] != 1 && error("`r` must start from 1")
    counts = zeros(length(r))
    for value in x
        if value <= r[end] && value >= 1
            counts[value] += 1
        end
    end
    return counts
end
