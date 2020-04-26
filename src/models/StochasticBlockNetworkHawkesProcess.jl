"""
An implementation of the network Hawkes model specified in Linderman, 2015.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `A::Array{Int32,2}`: binary matrix indicating network connections.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `μ::Array{Float64,2}`: matrix of impulse-response mean parameters.
- `τ::Array{Float64,2}`: matrix of impulse-response variance parameters.
- `Δtmax::Float64`: maximum lag parameter.
"""
mutable struct StochasticBlockNetworkHawkesProcess <: HawkesProcess
    # parameters
    λ0::Array{Float64,1}
    μ::Array{Float64,2}
    τ::Array{Float64,2}
    A::Array{Bool,2}
    W::Array{Float64,2}
    ρ::Array{Float64,2}
    z::Array{Int64,1}  # How to initialize?
    π::Array{Float64,1}
    Δtmax::Float64
    N::Int64
    # hyperparameters
    α0::Float64
    β0::Float64
    κ::Float64
    ν::Array{Float64,2}
    μμ::Float64
    κμ::Float64
    ατ::Float64
    βτ::Float64
    αρ::Float64
    βρ::Float64
    γ::Float64
    K::Int64
    function StochasticBlockNetworkHawkesProcess(λ0, μ, τ, A, W, ρ, z, π, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, αρ, βρ, γ, K)
        assert_nonnegative(λ0, "Intensity must be non-negative")
        assert_nonnegative(W, "Self-excitement strength must be non-negative")
        assert_nonnegative(τ, "Impulse-response variance must be non-negative")
        assert_nonnegative(Δtmax, "Maximum lag must be non-negative")
        # TODO: finish assertions
        return new(λ0, μ, τ, A, W, ρ, z, π, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, αρ, βρ, γ, K)
    end
end
