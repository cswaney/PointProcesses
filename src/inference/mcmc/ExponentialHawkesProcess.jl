"""
    mcmc(p::ExponentialHawkesProcess, data, params, n::Int64)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `nsamples::Int32`: number of samples to draw.
"""
function mcmc(process::ExponentialHawkesProcess, data, nsamples::Int64)
    @info "Running ExponentialHawkesProcess Gibbs sampler..."
    events, nodes, T = data
    λ0 = Array{typeof(process.λ0),1}(undef,nsamples)
    W = Array{typeof(process.W),1}(undef,nsamples)
    θ = Array{typeof(process.θ),1}(undef,nsamples)
    for i = 1:nsamples
        if i % 100 == 0
            @info "i=$i"
        end
        parents = sample_parents(process, events, nodes)
        parentnodes = [get_parent_node(nodes, p) for p in parents]
        λ0[i] = sample_baseline!(process, nodes, parentnodes, T)
        θ[i] = sample_impulse_response!(process, events, nodes, parents, parentnodes)
        W[i] = sample_weights!(process, events, nodes, parentnodes)
    end
    return λ0, W, θ
end


"""
    sample_impulse_response!(p::ExponentialHawkesProcess, events, nodes, parents, parentnodes)

The impulse-response is an exponential distribution. Assuming a conjugate prior `θ ~ Gamma(αθ, βθ)`, the posterior distribution of `θ[n, m]` is proportional to a `Gamma(α0 + Mnm, β + Mnm + Xnm)`, where `Mnm` is the number of events on node `m` with parent on node `n`, and `Xnm` is the average duration between these `Mnm` parent-child pairs. *Note*: In case that `Mnm = 0`, we default to the prior distribution (as we have observed no data related to `θ[n,m]`).
"""
function sample_impulse_response!(p::ExponentialHawkesProcess, events, nodes, parents, parentnodes)
    Mnm = parent_counts(nodes, parentnodes, p.N)
    Xnm = duration_mean(events, nodes, parents, p.N)
    αnm = p.αθ .+ Mnm  # = α (prior) for zero-counts
    βnm = p.βθ .+ Mnm .* Xnm  # Xnm returns 0 for zero-counts => β (prior)
    p.θ = rand.(Gamma.(αnm, 1 ./ βnm))
    return copy(p.θ)
end

function duration_mean(events, nodes, parents, size)
    Xnm = zeros(size, size)
    Mnm = zeros(size, size)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parentnode = nodes[parent]
            parentevent = events[parent]
            Mnm[parentnode, node] += 1
            Xnm[parentnode, node] += event - parentevent
        end
    end
    return fillna!(Xnm ./ Mnm, 0)
end


function sample_parent(p::ExponentialHawkesProcess, event, node, index, events, nodes)
    if index == 1
        return 0
    end
    λs = []
    parentindices = []
    ir = impulse_response(p)
    parentindex = 1
    while parentindex < index
        parenttime = events[parentindex]
        parentnode = nodes[parentindex]
        append!(λs, ir[parentnode, node](event - parenttime))
        append!(parentindices, parentindex)
        parentindex += 1
    end
    append!(λs, p.λ0[node])
    append!(parentindices, 0)
    return parentindices[argmax(rand(Discrete(λs ./ sum(λs))))]
end
