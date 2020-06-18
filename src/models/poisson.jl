using Distributions
import Base.rand


abstract type PoissonProcess end

"""
    intensity(p::PoissonProcess)

Construct the intensity function for a point process, `t -> λ(t)`.
"""
function intensity(p::PoissonProcess) end

"""
    likelihood(p::PoissonProcess)

Calculate the likelihood of a sequence of events over period `[0, T]`.
"""
function likelihood(p::PoissonProcess, events, T) end

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

function loglikelihood(p::HomogeneousProcess, ts, T)
    a = -p.λ * T
    b = length(ts) * log(p.λ)
    return a + b
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
    θ  # rate  = 1 / scale
end

intensity(p::ExponentialProcess) = t -> p.w * pdf(Exponential(p.θ), t)

function likelihood(p::ExponentialProcess, ts, T)
    h = Exponential(1 / p.θ)
    a = exp(-p.w * cdf(h, T))
    b = prod(p.w * pdf.(h, ts))
    return a * b
end

function loglikelihood(p::ExponentialProcess, ts, T)
    h = Exponential(1 / p.θ)
    a = -p.w * cdf(h, T)
    b = sum(log.(p.w * pdf.(h, ts)))
    return a + b
end

function rand(p::ExponentialProcess, T)
    h = Exponential(1 / p.θ)
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
    τ  # τ = 1 / σ^2 (precision)
    Δtmax
end

intensity(p::LogitNormalProcess) = t -> p.w * pdf(LogitNormal(p.μ, p.τ ^ (-1/2)), t / p.Δtmax)

function likelihood(p::LogitNormalProcess, ts, T)
    d = LogitNormal(p.μ, p.τ ^ (-1/2))
    a = exp(-p.w * cdf(d, T / p.Δtmax))
    b = prod(p.w * pdf.(d, ts ./ p.Δtmax))
    return a * b
end

function loglikelihood(p::LogitNormalProcess, ts, T)
    d = LogitNormal(p.μ, p.τ ^ (-1/2))
    a = -p.w * cdf(d, T / p.Δtmax)
    b = sum(log.(p.w * pdf.(d, ts ./ p.Δtmax)))
    return a + b
end

# logit(x) = log(x / (1 - x))
#
# function logit_normal(μ, τ, x)
#     Z = x * (1 - x) * (τ / (2 * π))^(-1/2)
#     return (1 / Z) * exp(-τ / 2 * ( logit(x) - μ)^2)
# end

function rand(p::LogitNormalProcess, T)
    T = max(p.Δtmax, T)
    h = LogitNormal(p.μ, p.τ ^ (-1/2))
    n = rand(Poisson(p.w * cdf(h, T / p.Δtmax)))
    ts = rand(truncated(h, 0., T / p.Δtmax), n)
    return sort(ts)
end


"""
A Poisson process with linear spline interpolation intensity function.
"""
struct LinearSplineProcess <: PoissonProcess
    t0::Float64
    Δt::Float64
    n::Int64  # xi = t0 + Δt * i, i = 0, ..., n
    ts::Array{Float64,1}
    λs::Array{Float64,1}  # λ0, λ1, ..., λn
    # hyperparameters
    α0::Float64
    β0::Float64
end

function LinearSplineProcess(t0, Δt, n, λs)
    ts = t0 .+ Δt .* 0:1:n
    α0 = β0 = 1.;
    return LinearSplineProcess(t0, Δt, n, ts, λs, α0, β0)
end

function intensity(p::LinearSplineProcess)
    function λ(t)
        i = min(Int(t ÷ p.Δt + 1), p.n)  # last interval is closed!
        t0 = p.ts[i]
        t1 = p.ts[i + 1]
        λ0 = p.λs[i]
        λ1 = p.λs[i + 1]
        return (λ1 * (t - t0) + λ0 * (t1 - t)) / p.Δt
    end
    return λ
end

function likelihood(p::LinearSplineProcess, ts, T)
    # integrated intensity
    I = 0.
    for i = 1:p.n
        t0 = p.ts[i]
        t1 = p.ts[i + 1]
        λ0 = p.λs[i]
        λ1 = p.λs[i + 1]
        I += p.Δt * (λ0 + 1/2 * (λ1 - λ0))
    end
    a = exp(-I)
    # product of intensity
    b = 1.
    for t in ts
        i = min(Int(t ÷ p.Δt + 1), p.n)
        t0 = p.ts[i]
        t1 = p.ts[i + 1]
        λ0 = p.λs[i]
        λ1 = p.λs[i + 1]
        b *= λ0 + (λ1 - λ0) * (t - t0) / p.Δt
    end
    return a * b
end

loglikelihood(p::LinearSplineProcess, ts, T) = log(likelihood(p, ts, T))

function rand(p::LinearSplineProcess, T)
    # integrated intensity
    I = 0.
    for i = 1:p.n
        t0 = p.ts[i]
        t1 = p.ts[i + 1]
        λ0 = p.λs[i]
        λ1 = p.λs[i + 1]
        I += p.Δt * (λ0 + 1/2 * (λ1 - λ0))
    end
    n = rand(Poisson(I))
    # temporal distribution
    ts = sample(p, n, T)
    return sort(ts)
    # events = []
    # for t in p.ts
    #     i = min(Int(t ÷ p.Δt + 1), p.n)
    #     t0 = p.ts[i]
    #     t1 = p.ts[i + 1]
    #     λ0 = p.λs[i]
    #     λ1 = p.λs[i + 1]
    #     I = p.Δt * (λ0 + 1/2 * (λ1 - λ0))
    #     n = rand(Poisson(I))
    #     append!(events, rand(Uniform(t0, t1), n))
    # end
    # return events
end

function sample(p::LinearSplineProcess, n, T)
    """Sample from the distribution defined by the linear spline using rejection-sampling (i.e. sample from 2-dimension rectangle containing spline graph and reject samples that lie above the graph).

    This method is slow, but accurate.
    """
    λ = intensity(p)
    # find the maximum λs => λmax
    λmax = maximum(p.λs)
    # sample uniformly over [0, T] x [0, λmax] => z
    U = Product([Uniform(0, T), Uniform(0, λmax)])
    # for each z, accept if z[2] < intensity(p)(z[1])
    s = []
    while length(s) < n
        z = rand(U, n)
        if z[2] < λ(z[1])
            push!(s, z[1])
        end
    end
    return s
end
