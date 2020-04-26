"""
    mcmc(p::StochasticBlockNetworkHawkes, data, nsamples::Int32)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `nsamples::Int32`: number of samples to draw.
"""
function mcmc(p::StochasticBlockNetworkHawkesProcess, data, nsamples::Int64)
    @info "Running StochasticBlockNetworkHawkesProcess Gibbs sampler..."
    events, nodes, T = data
    A = Array{typeof(p.A),1}(undef,nsamples)
    ρ = Array{typeof(p.ρ),1}(undef,nsamples)
    z = Array{typeof(p.z),1}(undef,nsamples)
    π = Array{typeof(p.π),1}(undef,nsamples)
    W = Array{typeof(p.W),1}(undef,nsamples)
    λ0 = Array{typeof(p.λ0),1}(undef,nsamples)
    μ = Array{typeof(p.μ),1}(undef,nsamples)
    τ = Array{typeof(p.τ),1}(undef,nsamples)
    for i = 1:nsamples
        # TODO: use ProgressBar.jl
        if i % 100 == 0
            @info "i=$i"
        end
        A[i] = sample_adjacency_matrix!(p, events, nodes, T)
        ρ[i] = sample_connection_probability!(p)
        z[i] = sample_classes!(p)
        π[i] = sample_class_probability!(p)
        parents = sample_parents(p, events, nodes)
        parentnodes = [get_parent_node(nodes, p) for p in parents]
        W[i] = sample_weights!(p, events, nodes, parentnodes)
        λ0[i] = sample_baseline!(p, nodes, parentnodes, T)
        μ[i], τ[i] = sample_impulse_response!(p, events, nodes, parents, parentnodes)
    end
    return λ0, μ, τ, W, A, ρ, z, π
end

function link_probability(p::StochasticBlockNetworkHawkesProcess)
    """Returns the probability of a connection between each pair of nodes."""
    L = ones(p.N, p.N)
    for i = 1:p.N
        for j = 1:p.N
            ki = p.z[i]
            kj = p.z[j]
            L[i, j] = p.ρ[ki, kj]
        end
    end
    return L
end

function sample_connection_probability!(p::StochasticBlockNetworkHawkesProcess)
    """Returns a sample from the conditional posterior of `ρ`. Equivalent to sampling prior for unobserved class combinations."""
    counts = zeros(p.K, p.K)
    connections = zeros(p.K, p.K)
    for i = 1:p.N
        for j = 1:p.N
            ki = p.z[i]
            kj = p.z[j]
            counts[ki, kj] += 1
            connections[ki, kj] += p.A[i, j]
        end
    end
    p.ρ = rand.(Beta.(p.αρ .+ connections, p.βρ .+ counts .- connections))
    return copy(p.ρ)
end

function sample_classes!(p::StochasticBlockNetworkHawkesProcess)
    """Return a sample from the conditional posterior distribution of latent classes."""
    products = [ones(p.K) for _ in 1:p.N]
    for i = 1:p.N
        for j = 1:p.N
            kn = p.z[j]
            products[i] .*= p.ρ[kn, :] .* p.ρ[:, kn]
        end
    end
    π_raw = [p.π .* prds for prds in products]
    z_raw = [sum(pr) for pr in π_raw]
    π = π_raw ./ z_raw
    p.z = dense(hcat(rand.(Discrete.(π))...))
    return copy(p.z)
end

function sample_class_probability!(p::StochasticBlockNetworkHawkesProcess)
    """Return a sample from the conditional posterior distribution of class probability."""
    p.π = rand(Dirichlet(p.γ .+ rangecount(p.z, 1:p.K)))
    return copy(p.π)
end

function rangecount(x::Array{Int64,1}, r::UnitRange{Int64})
    """
    Return the number of elements in `x` equal to each value in range `r`, which is assumed to start from 1.

    # Example
    x = [1, 3, 2, 2, 1, 4]
    r = 1:4
    rangecount(x, r)  # [2, 2, 1, 1]
    """
    r[1] != 1 && error("`r` must start from 1")
    counts = zeros(length(r))
    for value in x
        if value <= r[end] && value >= 1
            counts[value] += 1
        end
    end
    return counts
end
