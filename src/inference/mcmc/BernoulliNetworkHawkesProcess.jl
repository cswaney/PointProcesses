"""
    mcmc(p::BernoulliNetworkHawkesProcess, data, params, n::Int32)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `params`: tuple of hyperparameters
- `n::Int32`: number of samples to draw.
"""
function mcmc(p::BernoulliNetworkHawkesProcess, data, nsamples::Int64)
    @info "Running BernoulliNetworkHawkesProcess Gibbs sampler..."
    events, nodes, T = data
    λ0 = Array{typeof(p.λ0),1}(undef,nsamples)
    μ = Array{typeof(p.μ),1}(undef,nsamples)
    τ = Array{typeof(p.τ),1}(undef,nsamples)
    A = Array{typeof(p.A),1}(undef,nsamples)
    W = Array{typeof(p.W),1}(undef,nsamples)
    ρ = Array{typeof(p.ρ),1}(undef,nsamples)
    for i = 1:nsamples
        if i % 100 == 0
            @info "i=$i"
        end
        A[i] = sample_adjacency_matrix!(p, events, nodes, T)
        ρ[i] = sample_network!(p)
        parents = sample_parents(p, events, nodes)
        parentnodes = [get_parent_node(nodes, p) for p in parents]
        W[i] = sample_weights!(p, events, nodes, parentnodes)
        λ0[i] = sample_baseline!(p, nodes, parentnodes, T)
        μ[i], τ[i] = sample_impulse_response!(p, events, nodes, parents, parentnodes)
    end
    return λ0, μ, τ, W, A, ρ
end

link_probability(p::BernoulliNetworkHawkesProcess) = p.ρ * ones(p.N, p.N)

function sample_network!(p::BernoulliNetworkHawkesProcess)
    p.ρ = rand(Beta(p.αρ + sum(p.A), p.βρ + p.N * p.N - sum(p.A)))
    return copy(p.ρ)
end
