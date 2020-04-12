using Distributions
import Base.rand


abstract type PoissonProcess end

"""
rand(p::PoissonProcess, T::Float64)

Construct a random sample from a point process `p` on interval [0, `T`].

First, generate a number of events, `n`. Second, independently sample the time of each event using normalized intensity.
"""
function rand(p::PoissonProcess, T) end


"""
A homogeneous Poisson process.
"""
struct HomogeneousProcess <: PoissonProcess
    λ
end

intensity(p::HomogeneousProcess) = t -> p.λ
function likelihood(p::HomogeneousProcess, ts, T)
    a = exp(-p.λ * T)
    b = p.λ ^ length(ts)
    return a * b
end
function rand(p::HomogeneousProcess, T)
    n = rand(Poisson(p.λ * T))
    ts = rand(Uniform(0, T), n)
    return sort(ts)
end


"""
A Poisson process with linear intensity function.
"""
struct LinearProcess <: PoissonProcess
    b
    w
end

intensity(p::LinearProcess) = t -> p.b + p.w * t
function likelihood(p::LinearProcess, ts, T)
    @warn "TODO: assert that linear intensity is non-negative on [0, T]."
    a = exp(-(0.5 * p.m * T ^ 2 + p.b * T))
    b = prod(p.m .* ts .+ p.b)
    return a * b
end
function rand(p::LinearProcess, T)
    @error "Method not implemented"
end


"""
A Poisson process with Exponential distribution intensity function.
"""
struct ExponentialProcess <: PoissonProcess
    w
    θ
end

intensity(p::ExponentialProcess) = t -> p.w * pdf(Exponential(p.θ), t)
function likelihood(p::ExponentialProcess, ts, T)
    h = Exponential(p.θ)
    a = exp(-p.w * cdf(h, T))
    b = prod(p.w * pdf.(h, ts))
    return a * b
end
function rand(p::ExponentialProcess, T)
    h = Exponential(p.θ)
    n = rand(Poisson(p.w * cdf(h, T)))
    ts = rand(truncated(h, 0., T), n)
    return sort(ts)
end

"""
A Poisson process with LogitNormal distribution intensity function.
"""
struct LogitNormalProcess <: PoissonProcess
    w
    μ
    τ
    Δtmax
end

intensity(p::LogitNormalProcess) = t -> p.w * pdf(LogitNormal(p.μ, p.τ), t / p.Δtmax)
function likelihood(p::LogitNormalProcess, ts, T)
    d = LogitNormal(p.μ, p.τ)
    a = exp(-p.w * cdf(d, T / p.Δtmax))
    b = prod(p.w * pdf.(d, ts ./ p.Δtmax))
    return a * b
end
function rand(p::LogitNormalProcess, T)
    h = LogitNormal(p.μ, p.τ)
    n = rand(Poisson(p.w * cdf(h, T / p.Δtmax)))
    ts = rand(truncated(h, 0., T / p.Δtmax), n)
    return sort(ts)
end
