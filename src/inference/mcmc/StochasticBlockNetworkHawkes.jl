"""
    mcmc(p::StochasticBlockNetworkHawkes, data, params, n::Int32)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `params`: tuple of hyperparameters
- `n::Int32`: number of samples to draw.
"""
function mcmc(p::StochasticBlockNetworkHawkesProcess, data, nsamples::Int32)

    # latent variables - use process/network fields

    # hyperparameters - use process/network fields

    # Gibbs sampling
    _λ0_ = zeros((size(p.λ0)..., nsamples))
    _μ_ = zeros((size(p.μ)..., nsamples))
    _τ_ = zeros((size(p.τ)..., nsamples))
    _A_ = zeros((size(p.A)..., nsamples))
    _W_ = zeros((size(p.W)..., nsamples))
    _z_ = zeros((size(p.τ)..., nsamples))
    _π_ = zeros((size(p.π)..., nsamples))
    _ρ_ = zeros((size(p.ρ)..., nsamples))
    for iter = 1:nsamples
        parents = sample_parents(data)
        _λ0_[i] = sample_baseline!(p, data, parents)
        _μ_[i], _τ_[i] = sample_impulse_response!(p, data, parents)
        _W_[i] = sample_weights!(p, data, parents)
        _A_[i] = sample_adjacency!(p, data)
        _ρ_[i] = sample_connection_probability!(p)
        _z_[i] = sample_classes!(p)
        _π_[i] = sample_class_probability!(p)
    end
    return samples
end

function connection_likelihood(p::StochasticBlockNetworkHawkesProcess)
    L = ones(p.N, p.N)
    for i = 1:p.N
        for j = 1:p.N
            ki = p.z[i]
            kj = p.z[j]
            if p.A[i, j] == 1.
                L[i, j] = p.ρ[ki, kj]
            else
                L[i, j] = 1. - p.ρ[ki, kj]
        end
    end
    return L
end

function sample_connection_probability!(p::StochasticBlockNetworkHawkesProcess)
    K = length(p.γ)
    counts = zeros(K, K)
    connections = zeros(K, K)
    for i = 1:N
        for j = 1:N
            ki = p.z[i]
            kj = p.z[j]
            counts[ki, kj] += 1
            connections[ki, kj] += A[i, j]
        end
    end
    p.ρ = rand(Beta.(p.α .+ connections, p.β .+ counts .- connections))
    return copy(p.ρ)
end

function sample_classes!(p::StochasticBlockNetworkHawkesProcess)
    products = ones(length(p.π), p.N)
    for i = 1:p.N
        for j = 1:p.N
            kn = z[j]
            products[:, i] .*= ρ[kn, :] .* ρ[:, kn]
        end
    end
    p.z = rand(Discrete.(p.π .* products))
    return copy(p.z)
end

function sample_class_probability!(p::StochasticBlockNetworkHawkesProcess)
    p.π = rand(Dirichlet(p.γ + rangecount(z, 1:length(p.γ))))
    return copy(p.π)
end
