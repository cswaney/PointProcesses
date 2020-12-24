"""
Update the variational parameters specific to a Bernoulli network.

# Arguments
- ρ: variational parameter of a[i, j] ~ q(a[i, j] | ρ)
"""
function update_network(net::BernoulliNetwork, ρ)
    α, β = update_link_probabilities(net::Network, ρ)
    return α, β
end


function update_link_probabilities(net::BernoulliNetwork, ρ)
    α = net.α .+ net.N ^ 2 .* ρ
    β = net.β .+ net.N ^ 2 .* (1 .- ρ)
    return α, β
end


"""
Update the variational parameters specific to a stochastic block network.

# Arguments
- ρ: q(aij | ρ)
"""
function update_network(net::StochasticBlockNetwork, ρ)
    α, β = update_link_probabilities(net, ρ)  # q(ρ) = Beta(α, β)
    π = update_classes(net, α, β, π, γ)  # q(zn) = Mult(π)
    γ = update_class_probabilities(net, ρ, π)  # q(π) = Dir(γ)
    return α, β, π, γ
end


function update_link_probabilities(net::StochasticBlockNetwork, ρ, π)
    for i = 1:net.K
        for j = 1:net.K
            # α =
            # β =
        end
    end
    return α, β
end


function update_classes(net::StochasticBlockNetwork, α, β, π, γ)
    # TODO
end


function update_class_probabilities(net::StochasticBlockNetwork, ρ, π)
    # TODO
end
