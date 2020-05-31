mutable struct DiscreteHawkesProcess
    # parameters
    λ0::Array{Float64,1}
    θ::Array{Float64,3}
    W::Array{Float64,2}
    A::Array{Bool,2}
    Δt::Float64
    D::Int64
    B::Int64
    N::Int64
    # hyperparameters
    αλ::Float64
    βλ::Float64
    γ::Float64
    κ0::Float64
    ν0::Float64
    κ1::Float64
    ν1::Float64
    # network
    network::Network
end


"""
    basis(p::DiscreteHawkesProcess)

Calculate discretized Gaussian basis functions with means are evenly spaced along `[1, D]`.
"""
function basis(p::DiscreteHawkesProcess)
    D = p.D
    N, _, B = size(p.θ)
    σ = D / (B - 1)
    if D <= 2
        μ  = Array(LinRange(1, D, B + 2)[2:end - 1])
    else
        μ  = Array(LinRange(1, D, B))
    end
    lags = Array(1:D)
    ϕ = exp.(-1/2 * σ ^ -1/2 .* (lags .- transpose(μ)) .^ 2)
    # return ϕ ./ (sum(ϕ, dims=1) .* p.Δt)
    return [ϕ[:, b] ./ (sum(ϕ[:, b]) .* p.Δt) for b = 1:B]
end


"""
    rand(p::DiscreteHawkesProcess, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::DiscreteHawkesProcess`: model to sample from.
- `T::Int64`: end of sample period, `[0, T]`.
- `merged::Bool`: return sample as a `(times, nodes)` tuple.
"""
function rand(p::DiscreteHawkesProcess, T::Int64)
    N = length(p.λ0)
    D = p.D
    Δt = p.Δt
    λ0 = p.λ0
    IR = impulse_response(p)
    S = S0 = rand.(Poisson.(repeat(λ0 .* Δt, 1, T)))  # or rand.(Poisson.(λ0), T)
    for t = 1:T - 1
        # @info "t=$t"
        for n = 1:N
            # @info "n=$n"
            for s = 1:S[n, t]
                Dmax = min(D, T - t)
                λ = zeros(N, Dmax)
                # @info "Dmax=$Dmax"
                for d in 1:Dmax
                    λ[:, d] = p.A[n, :] .* p.W[n, :] .* IR[n, :, d]  # 1 x N x D
                end
                children = rand.(Poisson.(λ .* Δt))
                S[:, t + 1:t + Dmax] .+= children
                # @info "Generated $(sum(children)) events (t=$t, n=$n, s=$s)"
            end
        end
    end
    return S
end


"""
    augmented_loglikelihood(p::DiscreteHawkesProcess, parents, convolved)

Calculated the log-likelihood of the data given latent parent counts `parents`.

# Implementation Note
The `parents` array implicitly contains all information about events (summing across the last dimension gives the event count array).

# Arguments
- `events::Array{Int64,2}`: `N x T` array of event counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
- `parents::Array{Int64,3}`: `T x N x (1 + N * B)` array of parent counts.
"""
function augmented_loglikelihood(p::DiscreteHawkesProcess, parents, convolved)
    ll = 0.
    for t = 1:T
        for childchannel = 1:N
            λ = p.λ0[childchannel] * p.Δt
            ll += cdf(Poisson(λ), parents[t, childchannel, 1])
            for parentchannel = 1:N
                for b = 1:p.B
                    index = 1 + (parentchannel - 1) * p.B + b
                    shat = convolved[t, parentchannel, b]
                    a = p.A[parentchannel, childchannel]
                    w = p.W[parentchannel, childchannel]
                    θ = p.θ[parentchannel, childchannel, b]
                    λ = shat * a * w * θ * p.Δt
                    ll += pdf(Poisson(λ), parents[t, childchannel, index])
                end
            end
        end
    end
    return ll
end


"""
    loglikelihood(p::DiscreteNetworkHawkesProcess, events, convolved)

Calculate the log-likelihood of `events`.

# Implementation Note
The algorithm works by integrating/summing over parent counts. As the parent counts can only take values between zero and the number of events occuring at a given time on a given node, integration amounts to evaluating the cumulative density function.

# Arguments
- `events::Array{Int64,2}`: `N x T` array of event counts.
- `convolved::Array{Float64,3}`: `T x N x B` array of event counts convolved with basis functions.
"""
function loglikelihood(p::DiscreteHawkesProcess, events, convolved)
    ll = 0.
    _, T = size(events)
    for t = 1:T
        for childchannel = 1:p.N
            λ = p.λ0[childchannel] * p.Δt
            ll += cdf(Poisson(λ), events[childchannel, t])
            for parentchannel = 1:p.N
                for b = 1:p.B
                    shat = convolved[t, parentchannel, b]
                    a = p.A[parentchannel, childchannel]
                    w = p.W[parentchannel, childchannel]
                    θ = p.θ[parentchannel, childchannel, b]
                    λ = shat * a * w * θ * p.Δt
                    ll += cdf(Poisson(λ), events[childchannel, t])
                end
            end
        end
    end
    return ll
end


"""
    intensity(p::DiscreteHawkesProcess, events, convolved)

Calculate the intensity of `p` at time all times `t ∈ [1, 2, ..., T]` given `events` and pre-computed convolutions `convolved`.
"""
function intensity(p::DiscreteHawkesProcess, events, convolved)
    T, N, B = size(convolved)
    λ = zeros(T, N)
    for t = 1:T
        for childnode = 1:N
            λ[t, childnode] += p.λ0[childnode]
            for parentnode = 1:N
                for b = 1:B
                    # @info t, childnode, parentnode, b
                    shat = convolved[t, parentnode, b]
                    a = p.A[parentnode, childnode]
                    w = p.W[parentnode, childnode]
                    θ = p.θ[parentnode, childnode, b]
                    # TODO: check this calculation (paper switches from n to n')
                    λ[t, childnode] += a * w * θ * shat
                end
            end
        end
    end
    return λ
end


"""
    convolve(events, ϕ)

Convole each columns of `events` with each column (basis function) of `ϕ`.

Assumes `events` is `T x N` and `ϕ` is `D x B`. The convolution of each column of `ϕ` with `events` results in a `T x N` matrix. We stack up these matrices across all `B` columns of `ϕ`, resulting in a `T x N x B` array.
"""
function convolve(events, ϕ)
    N, T = size(events)
    D, _ = size(hcat(ϕ...))
    convolved = [conv(transpose(events), [0., u...])[1:T, :] for u in ϕ]
    convolved = cat(convolved..., dims=3)
    convolved[1, :, :] .= 0  # inexact for some reason...
    return convolved
end


"""
    impulse_response(p::DiscreteHawkesProcess)

Calculate the impulse-response of `p` for all parent-child-basis combinations.
"""
function impulse_response(p::DiscreteHawkesProcess)
    ϕ = hcat(basis(p)...)
    N = length(p.λ0)
    D = p.D
    IR = zeros(N, N, D)
    for n = 1:N
        IR[n, :, :] = p.θ[n, :, :] * transpose(ϕ)
    end
    return IR  # (N x N x B) x (B x D)= N x N x D
end


"""
    stability(p::DiscreteHawkesProcess)

Calculate the stability of process `p` (`p` is stable if the retun value is less than one).
"""
stability(p::DiscreteHawkesProcess) = maximum(abs.(p.A .* p.W))





mutable struct VariationalParameters
    α  # N
    β  # N
    γ  # (N x N) x B
    κ0  # N x N
    ν0  # N x N
    κ1  # N x N
    ν1  # N x N
    ρ  # N x N
    αρ  # N x N
    βρ  # N x N
    # u  # T x N
end


"""
Perform stochastic variational inference for a discrete-time process.

The variational distribution takes the form q(λ0)q(θ)q(W)q(A)q(ω), where:

    - q(λ0) = Gamma(α, β)
    - q(θ) = Dir(γ)
    - q(W) = Gamma(kappa , ν)
    - q(A) = Bern(ρ)
    - q(ω) = Mult(u)

# Arguments
- `n::Int64`: maximum number of updates to perform.

# Return
- α, β, γ, ν, κ, ρ
"""
function svi(p::DiscreteHawkesProcess, data::Array{Int64,2}, guess::VariationalParameters, n::Int64)
    # Pre-compute convolutions
    convolved = convolve(data, basis(process))
    # Unpack variational parameters
    u = 0
    α = guess.α
    β = guess.β
    γ = guess.γ
    κ0 = guess.κ0
    ν0 = guess.ν0
    κ1 = guess.κ1
    ν1 = guess.ν1
    ρ = guess.ρ
    αρ = guess.αρ
    βρ = guess.βρ

    _α_ = Array{typeof(α),1}(undef, n)
    _β_ = Array{typeof(β),1}(undef, n)
    _γ_ = Array{typeof(γ),1}(undef, n)
    _κ0_ = Array{typeof(κ0),1}(undef, n)
    _ν0_ = Array{typeof(ν0),1}(undef, n)
    _κ1_ = Array{typeof(κ1),1}(undef, n)
    _ν1_ = Array{typeof(ν1),1}(undef, n)
    _ρ_ = Array{typeof(ρ),1}(undef, n)
    _αρ_ = Array{typeof(αρ),1}(undef, n)
    _βρ_ = Array{typeof(βρ),1}(undef, n)
    # Perform optimization
    for i = 1:n
        u = update_parents(p, convolved, α, β, κ0, ν0, κ1, ν1, γ, ρ)
        α, β = update_background(p, data, u)
        _α_[i] = α
        _β_[i] = β
        γ = update_impulse_response(p, u)
        _γ_[i] = γ
        κ0, ν0, κ1, ν1 = update_weights(p, data, u)
        _κ0_[i] = κ0
        _ν0_[i] = ν0
        _κ1_[i] = κ1
        _ν1_[i] = ν1
        ρ = update_adjacency_matrix(p, αρ, βρ, ν0, κ0, ν1, κ1)
        _ρ_[i] = ρ
        αρ, βρ = update_network(p, ρ)
        _αρ_[i] = αρ
        _βρ_[i] = βρ
    end
    # Return estimates
    return _α_, _β_, _γ_, _κ0_, _ν0_, _κ1_, _ν1_, _ρ_, _αρ_, _βρ_
end


"""
Perform variational inference update on auxillary parent variables ("local context"). The required variational parameters are α, β, κ, ν, and γ.
"""
function update_parents(p::DiscreteHawkesProcess, convolved::Array{Float64,3}, α, β, κ0, ν0, κ1, ν1, γ, ρ)
    T, N, B = size(convolved)
    u = zeros(T, N, 1 + N * B)
    # TODO: parallelize over T?
    for tidx = 1:T
        for cidx = 1:N
            u[tidx, cidx, 1] = update_parent(0, cidx, nothing, α, β, κ0, ν0, κ1, ν1, γ, ρ)
            for pidx = 1:N
                start = 1 + (pidx - 1) * B + 1
                stop = start + B - 1
                shat = convolved[tidx, pidx, :]
                u[tidx, cidx, start:stop] = update_parent(pidx, cidx, shat, α, β, κ0, ν0, κ1, ν1, γ, ρ)
            end
        end
    end
    Z = sum(u, dims=3)
    return u ./ Z
end


# TODO: Confirm calculation of mixture distribution is correct.
# TODO: Consider whether calculation can be vectorized for all parent-child pairs.
"""
    update_parent(pidx, cidx, shat, α, β, κ1, ν1, κ0, ν0, γ, ρ)

Perform variational inference update on an individual auxillary parent.

"""
function update_parent(pidx, cidx, shat, α, β, κ0, ν0, κ1, ν1, γ, ρ)
    if pidx == 0
        Elogλ = digamma(α[cidx]) - log(β[cidx])
        return exp(Elogλ)
    else
        Elogθ = digamma.(γ[pidx, cidx, :]) .- digamma(sum(γ[pidx, cidx, :]))
        ElogW0 = digamma(κ0[pidx, cidx]) - log(ν0[pidx, cidx])
        ElogW1 = digamma(κ1[pidx, cidx]) - log(ν1[pidx, cidx])
        ElogW = (1 - ρ[pidx, cidx]) * ElogW0 + ρ[pidx, cidx] * ElogW1
        return shat .* exp.(Elogθ .+ ElogW)
    end
end


"""
    update_background(p::DiscreteHawkesProcess, data, u)

Perform variational inference update on the background rate parameters.

# Arguments
- `u::Array{Float,3}`: T x N x (1 + NB) variational parameter for the auxillary parent variables.
"""
function update_background(p::DiscreteHawkesProcess, data, u)
    N, T = size(data)
    α = p.αλ .+ sum(u[:, :, 1] .* transpose(data), dims=1)  # N x 1
    β = p.βλ .+ T .* p.Δt .* ones(N)
    return vec(α), β
end


"""
    update_impulse_response(p::DiscreteHawkesProcess, u)

Perform variational inference update on the impulse-response parameters.

# Arguments
- `u`: the variational parameter for the auxillary parent variables.
"""
function update_impulse_response(p::DiscreteHawkesProcess, u)
    N, _, B = size(p.θ)
    γ = zeros(N, N, B)
    counts = reshape(sum(u, dims=1), N, 1 + N * B)
    for pidx = 1:N
        for cidx = 1:N
            start = 1 + (pidx - 1) * B + 1
            stop = start + B - 1
            γnm = counts[cidx, start:stop]
            γ[pidx, cidx, :] = p.γ .+ γnm
        end
    end
    return γ
end


"""
    update_weights(p::DiscreteHawkesProcess, u)

Perform variational inference update on the weight parameters.

# Arguments
- `u`: the variational parameter for the auxillary parent variables.
"""
function update_weights(p::DiscreteHawkesProcess, data, u)
    _, N, B = size(p.θ)
    for pidx = 1:N
        for cidx = 1:N
            start = 1 + (pidx - 1) * B + 1
            stop = start + B - 1
            z = u[:, cidx, start:stop] .* transpose(data)[:, cidx]
            parent_counts = sum(z)
            event_counts = sum(data[cidx, :])
            κ0[pidx, cidx] = p.κ0 + parent_counts
            ν0[pidx, cidx] = p.ν0 + event_counts
            κ1[pidx, cidx] = p.κ1 + parent_counts
            ν1[pidx, cidx] = p.ν1 + event_counts
        end
    end
    return κ0, ν0, κ1, ν1
end


"""
    update_adjacency_matrix(p::DiscreteHawkesProcess, α, β, ν0, κ0, ν1, κ1)

Perform variational inference update on the adjacency matrix parameters.

# Implementation Noteρ
If ρ ~ Beta(α, β), then 1 - ρ ~ Beta(β, α), which implies that E[log (1 - ρ)] = digamma(β) - digamma(β + α).

# Arguments
- "Global" variational parameters `α, β, κ0, ν0, κ1`, and `ν1`.
"""
function update_adjacency_matrix(p::DiscreteHawkesProcess, αρ, βρ, ν0, κ0, ν1, κ1)
    logodds = (digamma.(αρ) .- digamma.(αρ .+ βρ)) .- (digamma.(βρ) .- digamma.(αρ .+ βρ))
    logodds .+= p.κ1 .* log.(p.ν1) .- loggamma.(p.κ1)
    logodds .+= loggamma.(κ1) .- κ1 .* log.(ν1)
    logodds .+= loggamma.(p.κ0) .- p.κ0 .* log.(p.ν0)
    logodds .+=  κ0 .* log.(ν0) .- loggamma.(κ0)
    logits = exp.(logodds)
    ρ = logits ./ (logits .+ 1)
    return ρ
end


"""
Update the variational parameters specific to a Bernoulli network.

# Arguments
- ρ: variational parameter of a[i, j] ~ q(a[i, j] | ρ)
"""
function update_network(p::DiscreteHawkesProcess, ρ)
    α, β = update_link_probabilities(p.network, ρ)
    return α, β
end


function update_link_probabilities(net::BernoulliNetwork, ρ)
    N = length(net.ρ)
    α = net.α .+ N ^ 2 .* ρ
    β = net.β .+ N ^ 2 .* (1 .- ρ)
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
