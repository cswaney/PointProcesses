# TODO: Modify to accomodate data with multiple series.

"""
sample_weights(events, nodes, parents, θ)

Sample from the conditional posterior distribution of parent weights.

- `events::Array{Float64}`: array of times in [0., T]
- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parents::Array{Int}`: array of nodes in {0, ..., N}
"""
function sample_weights(events, nodes, parent_nodes, θ)

    # sufficient statistics
    Mn = node_counts(nodes, θ.N)
    Mnm = parent_counts(nodes, parent_nodes, θ.N)

    # sample posterior
    κ = θ.κ .+ Mnm
    ν = θ.ν .+ Mn
    w = rand(Gamma(κ, ν))

    return w
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
    cnts = zeros(size, size)  # cnts[i, j] parent_node = i, child_node = j
    for (node, parent_node) in zip(nodes, parent_nodes)
        if parent_node > 0  # 0 = background event
            cnts[parent, node] += 1
        end
    end
    return cnts
end


"""
sample_baseline(events, nodes, parent_nodes, T, θ)

Sample from the conditional posterior distribution of baseline intensity, `λ`.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
- `T::Float64`: maximum of observation period
- `θ`: tuple storing model parameters
"""
function sample_baseline(nodes, parent_nodes, T, θ)

    # sufficient statistics
    M0 = baseline_counts(nodes, parent_nodes, θ.N)

    # sample posterior
    α = θ.α + M0
    β = θ.β + T
    λ = rand(Gamma(α, β))

    return λ
end


"""
baseline_counts(nodes, parent_nodes, size)

Count events on each node attributed to its baseline intensity.

- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
- `size::Int`: numer of network nodes
"""
function baseline_counts(nodes, parent_nodes, size)
    cnts = zeros(size)
    for (node, parent_node) in zip(nodes, parent_nodes)
        if parent == 0
            cnts[parent_node, node] += 1
        end
    end
    return cnts
end


"""
sample_impulse_response(events, nodes, parents, θ)

Sample from the conditional posterior distribution of impulse response parameters.

- `events::Array{Float64}`: array of times in [0., T]
- `nodes::Array{Int}`: array of nodes in {1, ..., N}
- `parents::Array{Int}`: array of indices indicating the parent event
- `parent_nodes::Array{Int}`: array of nodes in {0, ..., N}
"""
function sample_impulse_response(events, nodes, parents, parent_nodes, θ)

    # sufficient statistics
    Mnm = parent_counts(nodes, parent_nodes, θ.size))
    Xnm = log_duration_mean(events, nodes, parents, parent_nodes, θ.size, θ.tmax)
    Vnm = log_duration_volatility(Xnm, events, nodes, parents, θ.size, .tmax)

    #  sample posterior
    μnm = (θ.κμ .* θ.μμ .+ Mnm * Xnm) ./ (θ.κμ .+ Mnm)
    κnm = θ.κμ .+ Mnm
    αnm = θ.ατ .+ Mnm ./ 2
    βnm = Vnm ./ 2 .+ Mmn .* θ.κμ / (Mnm .+ θ.κμ) .* (Xnm .- θ.μμ).^2 ./2
    μ, τ = rand(NormalGamma(μnm, κnm, αnm, βnm))

    return μ, τ
end

ln(x) = log(x, exp(1))

function log_duration(parent, child, tmax)
    return ln((child - parent) / (tmax - (parent - child)))
end

function log_duration_mean(events, nodes, parents, parent_nodes, size, tmax)
    Xnm = zeros(size, size)
    Mnm = zeros(size, size)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = parent_nodes[parent]
            parent_event = events[parent]
            Mnm[parent_node, node] += 1
            Xnm[parent_node, node] += log_duration(parent_event, event, tmax)
        end
    end
    return Xnm ./ Mnm
end

function log_duration_volatility(Xnm, events, nodes, parents, parent_nodes, size, tmax)
    Vnm = zeros(size, size)
    for (event, node, parent) in zip(events, nodes, parents)
        if parent > 0
            parent_node = parent_nodes[parent]
            parent_event events[parent]
            Vnm[parent, node] += (log_duration(parent_event, event, tmax) - Xnm[parent_node, node])^2
        end
    end
    return Vnm
end


# TODO
"""
Sample the conditional posterior distribution of the adjacency matrix, `A`.
"""
function sample_adjacency(events, nodes, A, W, θ, ...) end


"""
sample_parents(events, nodes, tmax, λ0, H)

Sample the conditional posterior distribution of event parents.

- `events`: list of event times in `(0, T)`
- `nodes`: list of event nodes in `{1, ..., N}`
- `tmax::Float64`: maximum lag between parent and child event
- `λ0::Array{Float64,1}`: array of baseline intensities
- `H::Array{Function,2}`: matrix of impulse response functions
"""
function sample_parents(events, nodes, tmax, λ0, H)
    parent_events = []
    parent_nodes = []
    for (e, n) in zip(events, nodes)
        idx = events < e - tmax
        prior_events = append!([-1], events[idx])
        prior_nodes = append!([0], nodes[idx])
        pe, pn = sample_parent(prior_events, prior_nodes, e, n, tmax, λ0, H)
        append!(parent_events, pe)
        append!(parent_nodes, pn)
    end
    return parent_events, parent_nodes
end

"""
sample_parent(event, node, events, nodes, tmax, λ0, H)

Sample the conditional posterior distribution of a single event parent.

- `event::Float64`: time of event in `(0, T)`
- `nodes`: node of event in `{1, ..., N}`
- `events`: list of event times in `(0, T)`
- `nodes`: list of event nodes in `{1, ..., N}`
- `tmax::Float64`: maximum lag between parent and child event
- `λ0::Array{Float64,1}`: array of baseline intensities
- `H::Array{Function,2}`: matrix of impulse response functions
"""
function sample_parent(event, node, events, nodes, tmax, λ0, H)
    λs = [λ0]
    for (e, n) in zip(events, nodes)
        Δt = event - e
        append!(λs, H[n, node](Δt))
    end
    idx = rand(Multinomial(1, λs / sum(λs)))
    return events[idx], nodes[idx]
end


"""
mcmc(...)

Run MCMC algorithm for inference in network Hawkes model.
"""
function mcmc(events, nodes, tmax, T, nsamples)

    # set constants

    # set initial estimates—probably these should be passed in or computed intelligently somehow...
    λ0 = zeros(N)

    # θ = (λ0, ...)

    # create arrays to hold samples # TODO: define correctly to store parameters...
    baseline_history = Array{Float64,2}(undef, nsamples, N)
    weights_history = Array{Float64,3}(undef, nsamples, N, N)
    impulse_response_history = Array{Float64,2}(undef, nsamples, N)
    adjacency_history = Array{Float64,2}(undef, nsamples, N, N)

    # sample conditional posterior (Gibbs)
    for i in 1:nsamples
        # sample parents
        # impulse_response = (convert impulse response distribution to functions on (0, tmax))
        parent_events, parent_nodes = sample_parents(events, nodes, tmax, λ0, impulse_response)

        # sample baseline
        baseline = sample_baseline(nodes, parent_nodes, T, θ)

        # sample weights
        weights = sample_weights(events, nodes, parent_nodes, θ)

        # sample impulse response
        impulse_response = sample_impulse_response(events, nodes, parents, parent_nodes, θ)

        # sample adjacency matrix
        # TODO adjacency_matrix = sample_adjacency_matrix(...)

        # store samples
        baseline_history[i, :] = baseline
        weights_history[i, :, :] = weights
        impulse_response_history[i, :, :] = impulse_response
        adjacency_matrix_history[i, :, :] = adjacency_matrix
    end

    return baseline_history, weights_history, impulse_response_history, adjacency_matrix_history
end
