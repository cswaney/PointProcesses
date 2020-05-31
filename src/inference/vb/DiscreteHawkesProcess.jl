"""Method to implement variational inference for discrete-time Hawkes processes."""

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
end


"""
Perform mean-field variational Bayesian inference for a discrete-time process.

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
function variational_bayes(p::DiscreteHawkesProcess, data::Array{Int64,2}, guess::VariationalParameters, n::Int64)
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
