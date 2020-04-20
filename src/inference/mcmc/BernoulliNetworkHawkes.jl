"""
    mcmc(p::BernoulliNetworkHawkesProcess, data, params, n::Int32)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `params`: tuple of hyperparameters
- `n::Int32`: number of samples to draw.
"""
function mcmc(p::BernoulliNetworkHawkesProcess, data, nsamples::Int32)
    _A_ = zeros((size(p.A)..., nsamples))
    _W_ = zeros((size(p.W)..., nsamples))
    _λ0_ = zeros((size(p.λ0)..., nsamples))
    _μ_ = zeros((size(p.μ)..., nsamples))
    _τ_ = zeros((size(p.τ)..., nsamples))
    _ρ_ = zeros((size(p.ρ)..., nsamples))
    for iter = 1:nsamples
        parents = sample_parents(data)
        _W_[i] = sample_weights!(p, data, parents)
        _λ0_[i] = sample_baseline!(p, data, parents)
        _μ_[i], _τ_[i] = sample_impulse_response!(p, data, parents)
        _A_[i] = sample_adjacency!(p, data)
        _ρ_[i] = sample_network!(p)
    end
    return samples
end

link_probability(p::BernoulliNetworkHawkesProcess) = p.ρ * ones(p.N, p.N)

function sample_network!(p::BernoulliNetworkHawkesProcess)
    p.ρ = rand(Beta(p.αρ + sum(p.A), p.βρ + sum(p.A) - p.N * p.N))
    return copy(p.ρ)
end
