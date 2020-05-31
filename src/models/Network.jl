abstract type Network end


"""
A fully-connected network.
"""
mutable struct DenseNetwork <: Network
    N
end


link_probability(net::DenseNetwork) = ones(net.N, net.N)


"""
A network model with independent Bernoulli distributed link probabilities.
"""
mutable struct BernoulliNetwork <: Network
    ρ
    α::Float64
    β::Float64
    N
end


link_probability(net::BernoulliNetwork) = net.ρ * ones(net.N, net.N)


"""
A network model with class-conditional Bernoulli disstributed link probabilities.
"""
mutable struct StochasticBlockNetwork <: Network
    ρ
    z
    π
    α::Float64
    β::Float64
    γ::Float64
    N
    K
end


function link_probability(net::StochasticBlockNetwork)
    """Returns the probability of a connection between each pair of nodes."""
    L = ones(net.N, net.N)
    for i = 1:net.N
        for j = 1:net.N
            ki = net.z[i]
            kj = net.z[j]
            L[i, j] = net.ρ[ki, kj]
        end
    end
    return L
end
