"""Methods to implement Gibbs samplingn on discrete-time Hawkes process."""


"""
    sample_parents(events, convolved, total_intensity)

# TODO: vectorize?

# Arguments
- `events::Array{Int64,2}`: `N x T` array of event counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
- `total_intensity::Array{Float64,2}`: `T x N` array of total intensity.
"""
function sample_parents(p::DiscreteHawkesProcess, events, convolved)
    T, N, B = size(convolved)
    parents = zeros(T, N, 1 + N * B)
    for t = 1:T
        for childnode = 1:N
            λ0 = p.λ0[childnode]
            λ = zeros(B, N)
            s = events[childnode, t]
            for parentnode = 1:N
                a = p.A[parentnode, childnode]
                w = p.W[parentnode, childnode]
                for b = 1:B
                    shat = convolved[t, parentnode, b]
                    θ = p.θ[parentnode, childnode, b]
                    λ[b, parentnode] = shat * a * w * θ
                end
            end
            π = [λ0, vec(λ)...] / sum([λ0, vec(λ)...])
            parents[t, childnode, :] = rand(Multinomial(s, π))
        end
    end
    return parents
end

function psample_parents(p::DiscreteHawkesProcess, events, convolved)
    T, N, B = size(convolved)
    parents = SharedArray(zeros(Int64, T, N, 1 + N * B))
    @sync @distributed for t = 1:T
        parents[t, :, :] = psample_parent(events[:, t], convolved[t, :, :], p.λ0, p.A, p.W, p.θ)
    end
    return parents
end

function psample_parent(events, convolved, λ0, A, W, θ)
    N, B = size(convolved)
    parents = zeros(Int64, N, 1 + N * B)
    for childnode = 1:N
        λn0 = λ0[childnode]
        Λ = zeros(B, N)
        s = events[childnode]
        for parentnode = 1:N
            Anm = A[parentnode, childnode]
            Wnm = W[parentnode, childnode]
            for b = 1:B
                shat = convolved[parentnode, b]
                θnm = θ[parentnode, childnode, b]
                Λ[b, parentnode] = shat * Anm * Wnm * θnm
            end
        end
        π = [λn0, vec(Λ)...] / sum([λn0, vec(Λ)...])
        parents[childnode, :] = rand(Multinomial(s, π))
    end
    return parents
end


"""
    sample_background(p::DiscreteHawkesProcess, parents, T)

# Arguments
- `parents::Array{Int64,3}`: `T x N x (1 + N * B)` array of parent counts.
"""
function sample_background(p::DiscreteHawkesProcess, parents)
    T, _, _ = size(parents)
    α = p.αλ .+ sum(parents[:, :, 1], dims=1)
    β = p.βλ + T * p.Δt
    p.λ0 = vec(rand.(Gamma.(α, 1 / β)))
    return copy(p.λ0)
end


"""
    sample_weights(p::DiscreteHawkesProcess, events, parents)

# Arguments
- `events::Array{Int64,2}`: `N x T` array of event counts.
- `parents::Array{Int64,3}`: `T x N x (1 + N * B)` array of parent counts.
"""
function sample_weights(p::DiscreteHawkesProcess, events, parents)
    κ = p.κ .+ sum_parents(p, parents)
    ν = p.ν .+ sum(events, dims=2)
    p.W = rand.(Gamma.(κ, 1 ./ ν))
    return copy(p.W)
end

function sum_parents(p::DiscreteHawkesProcess, parents)
    counts = zeros(p.N, p.N)
    for parentchannel = 1:p.N
        for childchannel = 1:p.N
            start = 1 + (parentchannel - 1) * p.B + 1
            stop = start + p.B - 1
            counts[parentchannel, childchannel] = sum(parents[:, childchannel, start:stop])
        end
    end
    return counts
end


"""
    sample_impulse_response(p::DiscreteNetworkHawkesProcess, parents)

# Arguments
- `parents::Array{Int64,3}`: `T x N x (1 + N * B)` array of parent counts.
"""
function sample_impulse_response(p::DiscreteHawkesProcess, parents)
    γ = Array{Array{Int64,1},2}(undef, p.N, p.N)
    counts = reshape(sum(parents, dims=1), p.N, 1 + p.N * p.B)
    for parentchannel = 1:p.N
        for childchannel = 1:p.N
            start = 1 + (parentchannel - 1) * p.B + 1
            stop = start + p.B - 1
            γnm = counts[childchannel, start:stop]
            γ[parentchannel, childchannel] = p.γ .+ γnm
        end
    end
    θ = rand.(Dirichlet.(γ))
    p.θ = reshape(transpose(cat(θ..., dims=2)), (p.N, p.N, p.B))
    return copy(p.θ)
end


"""
    sample_adjacency_matrix!(p::DiscreteHawkesProcess, events, T)

Sample the conditional posterior distribution of the adjacency matrix, `A`.
"""
function sample_adjacency_matrix!(p::DiscreteHawkesProcess, events, convolved)
    L = link_probability(p)
    for cidx = 1:p.N  # TODO: parallelize over `cidx` (columns of A)
        for pidx = 1:p.N
            # Set A[pidx, cidx] = 0
            ll0 = conditional_loglikelihood(p, events, convolved, 0, pidx, cidx)
            ll0 += log(1 - L[pidx, cidx])
            # Set A[pidx, cidx] = 1
            ll1 = conditional_loglikelihood(p, events, convolved, 1, pidx, cidx)
            ll1 += log(L[pidx, cidx])
            # Sample A[pidx, cidx]
            lZ = logsumexp(ll0, ll1)
            ρ = exp(ll1 - lZ)
            p.A[pidx, cidx] = rand.(Bernoulli.(ρ))
        end
    end
    return copy(p.A)
end

function psample_adjacency_matrix!(p::DiscreteHawkesProcess, events, convolved)

    # parallel
    A = SharedArray(zeros(Int64, p.N, p.N))
    link_prob = link_probability(p)
    @sync @distributed for column = 1:p.N
        A[:, column] = psample_column(events, convolved, column, link_prob, p.λ0, p.A, p.W, p.θ, p.Δt)
    end
    p.A .= A  # set element-wise

    # parallel
    # columns = pmap(sample_column(p, events, convolved, column), 1:p.N)

    # standard
    # for column = 1:p.N
    #     p.A[:, column] = sample_column(p, events, convolved, column)
    # end

    return copy(p.A)
end

function sample_column(p::DiscreteHawkesProcess, events, convolved, column)
    L = link_probability(p)
    a = zeros(p.N)
    for row = 1:p.N
        # Set A[row, cidx] = 0
        ll0 = conditional_loglikelihood(p, events, convolved, 0, row, column)
        ll0 += log(1 - L[row, column])
        # Set A[row, column] = 1
        ll1 = conditional_loglikelihood(p, events, convolved, 1, row, column)
        ll1 += log(L[row, column])
        # Sample A[row, column]
        lZ = logsumexp(ll0, ll1)
        ρ = exp(ll1 - lZ)
        a[row] = rand.(Bernoulli.(ρ))
    end
    return a
end

function psample_column(events, convolved, column, link_prob, λ0, A, W, θ, Δt)
    N, _ = size(events)
    a = zeros(N)
    for row = 1:N
        # Set A[row, cidx] = 0
        ll0 = pconditional_loglikelihood(events, convolved, 0, row, column, λ0, A, W, θ, Δt)
        ll0 += log(1 - link_prob[row, column])
        # Set A[row, column] = 1
        ll1 = pconditional_loglikelihood(events, convolved, 1, row, column, λ0, A, W, θ, Δt)
        ll1 += log(link_prob[row, column])
        # Sample A[row, column]
        lZ = logsumexp(ll0, ll1)
        ρ = exp(ll1 - lZ)
        a[row] = rand.(Bernoulli.(ρ))
    end
    return a
end

function conditional_loglikelihood(p::DiscreteHawkesProcess, events, convolved, a, pidx, cidx)
    ll = 0.
    _, T = size(events)
    for t = 1:T
        λ = p.λ0[cidx] * p.Δt
        ll += cdf(Poisson(λ), events[cidx, t])
        for parentchannel = 1:p.N
            for b = 1:p.B
                shat = convolved[t, parentchannel, b]
                w = p.W[parentchannel, cidx]
                θ = p.θ[parentchannel, cidx, b]
                if parentchannel == pidx
                    λ = shat * a * w * θ * p.Δt
                else
                    λ = shat * p.A[parentchannel, cidx] * w * θ * p.Δt
                end
                ll += cdf(Poisson(λ), events[cidx, t])
            end
        end
    end
    return ll
end

function pconditional_loglikelihood(events, convolved, a, pidx, cidx, λ0, A, W, θ, Δt)
    ll = 0.
    N, T = size(events)
    _, _, B = size(θ)
    for t = 1:T
        λ = λ0[cidx] * Δt
        ll += cdf(Poisson(λ), events[cidx, t])
        for parentchannel = 1:N
            for b = 1:B
                shat = convolved[t, parentchannel, b]
                Wnm = W[parentchannel, cidx]
                θnmb = θ[parentchannel, cidx, b]
                if parentchannel == pidx
                    λ = shat * a * Wnm * θnmb * Δt
                else
                    λ = shat * A[parentchannel, cidx] * Wnm * θnmb * Δt
                end
                ll += cdf(Poisson(λ), events[cidx, t])
            end
        end
    end
    return ll
end


"""
    mcmc!(p::DiscreteHawkesProcess, nsamples)

Generate `nsamples` MCMC samples from `p`. Overwrites parameters of `p`.

# Returns
- List of parameter samples.
"""
function mcmc(p::DiscreteHawkesProcess, events, nsamples)
    ϕ = basis(p)
    convolved = convolve(events, ϕ)
    λ0 = Array{typeof(p.λ0),1}(undef,nsamples)
    θ = Array{typeof(p.θ),1}(undef,nsamples)
    A = Array{typeof(p.A),1}(undef,nsamples)
    ρ = Array{typeof(p.ρ),1}(undef,nsamples)
    z = Array{typeof(p.z),1}(undef,nsamples)
    π = Array{typeof(p.π),1}(undef,nsamples)
    W = Array{typeof(p.W),1}(undef,nsamples)
    for iter in 1:nsamples
        iter % 100 == 0 && @info "iter=$iter"
        parents = psample_parents(p, events, convolved)
        λ0[iter] = sample_background(p, parents)
        θ[iter] = sample_impulse_response(p, parents)
        W[iter] = sample_weights(p, events, parents)
        if typeof(p.net) == StochasticBlockNetwork
            A[iter] = psample_adjacency_matrix!(p, events, convolved)
            ρ[iter] = sample_connection_probability!(p, A)
            z[iter] = sample_classes!(p)
            π[iter] = sample_class_probability!(p)
        elseif typeof(p.net) == BernoulliNetwork
            A[iter] = psample_adjacency_matrix!(p, events, convolved)
            ρ[iter] = sample_connection_probability!(p, A)
        end
    end
    if typeof(p.net) == StochasticBlockNetwork
        return λ0, θ, A, ρ, z, π, W
    elseif typeof(p.net) == BernoulliNetwork
        return λ0, θ, A, ρ, W
    else
        return λ0, θ, W
    end
end
