"""
    mcmc(p::NetworkHawkesProcess, data, params, n::Int64)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `params`: tuple of hyperparameters
- `n::Int32`: number of samples to draw.
"""
function mcmc(process::NetworkHawkesProcess, data, nsamples::Int64)
    events, nodes, T = data
    _λ0_ = Array{typeof(process.λ0),1}(undef,nsamples)
    _μ_ = Array{typeof(process.μ),1}(undef,nsamples)
    _τ_ = Array{typeof(process.τ),1}(undef,nsamples)
    _W_ = Array{typeof(process.W),1}(undef,nsamples)
    for i = 1:nsamples
        if i % 100 == 0
            @info "i=$i"
        end
        parents = sample_parents(process, events, nodes)
        parentnodes = [get_parent_node(nodes, p) for p in parents]
        _λ0_[i] = sample_baseline!(process, nodes, parentnodes, T)
        _μ_[i], _τ_[i] = sample_impulse_response!(process, events, nodes, parents, parentnodes)
        _W_[i] = sample_weights!(process, events, nodes, parentnodes)
    end
    return _λ0_, _μ_, _τ_, _W_
end
