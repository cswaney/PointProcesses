using Distributions, Random
import Random.GLOBAL_RNG

dense(onehots::Array{Int,2}) = vec([idx.I[1] for idx in argmax(onehots, dims=1)])
Discrete(π) = Multinomial(1, π)

struct NormalGammaSampler <: Sampleable{Multivariate,Continuous}
    μ
    λ
    α
    β  # β = 1 / θ (θ = "rate")
end

import Base.length
length(s::NormalGammaSampler) = 2

import Distributions._rand!
function _rand!(rng::AbstractRNG, s::NormalGammaSampler, x::AbstractVector{<:Real})
    T = rand(Gamma(s.α, 1 / s.β))  # Distributions.Gamma uses rate (θ = 1 / β)
    V = 1 / (s.λ * T)
    z = rand(Normal(s.μ, sqrt(V)))
    x[1] = z
    x[2] = T
    return x
end

_rand!(s::NormalGammaSampler, x::AbstractVector{<:Real}) = _rand!(GLOBAL_RNG, s, x)


struct NormalGamma <: ContinuousMultivariateDistribution
    μ
    λ
    α
    β
end

length(s::NormalGamma) = 2

import Distributions.sampler
sampler(s::NormalGamma) = NormalGammaSampler(s.μ, s.λ, s.α, s.β)

_rand!(rng::AbstractRNG, s::NormalGamma, x::AbstractVector{<:Real}) = _rand!(sampler(s), x)
_rand!(s::NormalGamma, x::AbstractVector{<:Real}) = _rand!(GLOBAL_RNG, s, x)
