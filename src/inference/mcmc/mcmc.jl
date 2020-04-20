# TODO: Modify to accomodate data with multiple series.

"""
sample_weights(events, nodes, parents, θ)

Sample from the conditional posterior distribution of parent weights.

- `events::Array{Float64}`: array of times in [0., T]
- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parents::Array{Int}`: array of events in {0, ..., M}
"""
function sample_weights!(p::HawkesProcess, events, nodes, parent_nodes)
    Mn = node_counts(nodes, p.N)
    Mnm = parent_counts(nodes, parent_nodes, p.N)
    κ = p.κ .+ Mnm
    ν = p.ν .+ Mn
    p.W = rand.(Gamma.(κ, 1 ./ ν))
    return copy(p.W)
end

"""
node_counts(nodes, size)

Count events on each node.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `size::Int`: number of network nodes
"""
function node_counts(nodes, size)
    cnts = zeros(size)
    for node in nodes
        cnts[node] += 1
    end
    return cnts
end

"""
parent_counts(nodes, parents, size)

Count events on each node attributed to each other node.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
- `size::Int`: number of network nodes
"""
function parent_counts(nodes, parent_nodes, size)
    cnts = zeros(size, size)
    for (node, parent_node) in zip(nodes, parent_nodes)
        if parent_node > 0  # 0 => background event
            cnts[parent_node, node] += 1
        end
    end
    return cnts
end

function get_parent_node(nodes, parent)
    if parent == 0
        return 0
    else
        return nodes[parent]
    end
end

function get_parent_event(events, parent)
    if parent == 0
        return -1
    else
        return events[parent]
    end
end


"""
sample_baseline(events, nodes, parent_nodes, T, θ)

Sample from the conditional posterior distribution of baseline intensity, `λ`.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
- `T::Float64`: maximum of observation period
- `θ`: tuple storing model parameters
"""
function sample_baseline!(p::HawkesProcess, nodes, parentnodes, T)
    M0 = baseline_counts(nodes, parentnodes, p.N)
    α = p.α0 .+ M0
    β = p.β0 .+ T  # (rate)
    p.λ0 = rand.(Gamma.(α, 1 ./ β))  # θ = 1 / β (scale)
    return copy(p.λ0)
end


"""
baseline_counts(nodes, parent_nodes, size)

Count events on each node attributed to its baseline intensity.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
- `size::Int`: numer of network nodes
"""
function baseline_counts(nodes, parentnodes, size)
    cnts = zeros(size)
    for (node, parentnode) in zip(nodes, parentnodes)
        if parentnode == 0
            cnts[node] += 1
        end
    end
    return cnts
end


"""
    sample_impulse_response(events, nodes, parents, θ)

Sample from the conditional posterior distribution of impulse response parameters.

# Implementation Notes
If `A[i,j] == 0` or `W[i,j] ≈ 0`, there will be no events on node `j` with latent parent on node `i`. As a result, `Xmn[i, j] == NaN`. In this case, we replace the parameters for the conditional distributions of `τ` and `μ` with the hyperparameters of the priors on these parameters. Intuitively, the data doesn't contain information about `τ` or `μ` in this case, so we should sample based on our prior beliefs.

# Arugments
- `events::Array{Float64}`: array of times in [0., T]
- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parents::Array{Int}`: array of indices indicating the parent event
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
"""
function sample_impulse_response!(p::HawkesProcess, events, nodes, parents, parentnodes)
    Mnm = parent_counts(nodes, parentnodes, p.N)
    Xnm = log_duration_mean(events, nodes, parents, p.N, p.Δtmax)
    if any(isnan.(Xnm))
        @warn "`Xnm` contains `NaN`s. This typically means that there were no parent-child observations for some combination of nodes."
    end
    Vnm = log_duration_volatility(Xnm, events, nodes, parents, p.N, p.Δtmax)
    μnm = fillna!((p.κμ .* p.μμ .+ Mnm .* Xnm) ./ (p.κμ .+ Mnm), p.μμ)  # use prior if no observation on connection
    κnm = p.κμ .+ Mnm  # NOTE: equals κμ if no obs. on connection
    αnm = p.ατ .+ Mnm ./ 2  # NOTE: equals κμ if no obs. on connection
    βnm = fillna!(Vnm ./ 2 .+ Mnm .* p.κμ ./ (Mnm .+ p.κμ) .* (Xnm .- p.μμ).^2 ./2, p.βτ)
    p.τ = rand.(Gamma.(αnm, 1 ./ βnm))
    σ = (1 ./ (κnm * p.τ)) .^ (1 / 2)
    p.μ = rand.(Normal.(μnm, σ))
    return copy(p.μ), copy(p.τ)
end

function fillna!(X, value)
    for index in eachindex(X)
        if isnan(X[index])
            X[index] = value
        end
    end
    return X
end

function log_duration(parent, child, Δtmax)
    return log((child - parent) / (Δtmax - (child - parent)))
end

function log_duration_mean(events, nodes, parents, size, Δtmax)
    Xnm = zeros(size, size)
    Mnm = zeros(size, size)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = nodes[parent]
            parent_event = events[parent]
            Mnm[parent_node, node] += 1
            Xnm[parent_node, node] += log_duration(parent_event, event, Δtmax)
        end
    end
    return Xnm ./ Mnm
end

function log_duration_volatility(Xnm, events, nodes, parents, size, Δtmax)
    M = length(events)
    Vnm = zeros(size, size)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = nodes[parent]
            parent_event = events[parent]
            Vnm[parent_node, node] += (log_duration(parent_event, event, Δtmax) - Xnm[parent_node, node])^2
        end
    end
    return Vnm
end


"""
sample_adjacency_matrix!(p::HawkesProcess, events, nodes, parents, T)

    Sample the conditional posterior distribution of the adjacency matrix, `A`.
"""
function sample_adjacency_matrix!(process::HawkesProcess, events, nodes, parents, T)
    L = link_probability(process)
    I = hcat([intensity(process, events, nodes, t0) for t0 in events]...)  # N x M
    Mn = node_counts(nodes, p.N)
    for i = 1:processs.N
        integral = process.λ0 * T + sum(p.A[:, i] .* process.W[:, i] .* Mn)
        idx = nodes[nodes == i]
        product = prod(I[i, idx])
        process.A[:, i] = rand(Bernoulli(integral .* product .* L[:, i]))
    end
    return copy(process.A)
end


"""
sample_parents(process::HawkesProcess, events, nodes)

    Sample the conditional posterior distribution of event parents.

    - `events`: list of event times in `(0, T)`
    - `nodes`: list of event nodes in `{1, ..., N}`
    - `tmax::Float64`: maximum lag between parent and child event
    - `λ0::Array{Float64,1}`: array of baseline intensities
    - `H::Array{Function,2}`: matrix of impulse response functions
"""
function sample_parents(process::HawkesProcess, events, nodes)
    parents = []
    for (index, (event, node)) in enumerate(zip(events, nodes))
        parent = sample_parent(process, event, node, index, events, nodes)
        append!(parents, parent)
    end
    return parents
end

function sample_parent(p::HawkesProcess, event, node, index, events, nodes)
    if index == 1
        return 0
    end
    λs = []
    parentindices = []
    ir = impulse_response(p)
    parentindex = index - 1
    while events[parentindex] > event - p.Δtmax
        parenttime = events[parentindex]
        parentnode = nodes[parentindex]
        if p.A[parentnode, node] == 1
            append!(λs, ir[parentnode, node](event - parenttime))
            append!(parentindices, parentindex)
            parentindex -= 1
            parentindex == 0 && break
        end
    end
    append!(λs, p.λ0[node])
    append!(parentindices, 0)
    return parentindices[argmax(rand(Discrete(λs ./ sum(λs))))]
end


"""
    mcmc(p::HawkesProcess, data, params, n::Int32)

Run MCMC algorithm for inference in network Hawkes model.

# Implementation Notes
We use the process and network structures to update model parameters during Gibbs sampling. Each `sample` method modifies its associated model parameters in the process or network and returns the sampled value. (The exception is the `sample_parents` method, which samples a latent variable).

- `data`: (events, nodes, T) tuple
- `params`: tuple of hyperparameters
- `n::Int32`: number of samples to draw.
"""
function mcmc(p::HawkesProcess, data, nsamples::Int64) end
