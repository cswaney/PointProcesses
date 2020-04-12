using Distributions
import Base.rand


abstract type Integrable end

"""
A univariate, piecewise linear function.

The function is defined by `n + 1` endpoints and `n` slopes.
"""
struct UnivariateLinear <: Integrable
    xs
    ms
    b0
    function UnivariateLinear(b0, xs, ms)
        if length(xs) != length(ms) + 1
            @error "Length mismatch"
        end
        return new(xs, ms, b0)
    end
end


"""
integrate(f::UnivariateLinear)

Integrate `f` over its domain.
"""
function integrate(f::UnivariateLinear)
    res = 0.
    b = f.b0
    F(x, m, b) = 0.5 * m * x ^ 2 + b * x
    for (i, x) in enumerate(f.xs[1:end - 1])
        m = f.ms[i]
        xnext = f.xs[i + 1]
        res += F(xnext - x, m, b) - F(0, m, b)
        @info "x=$x, xnext=$xnext, b=$b, m=$m"
        b += (xnext - x) * m
    end
    return res
end


function evaluate(f::UnivariateLinear, x0)
    if x0 > f.xs[end] || x0 < f.xs[1]
        @error "`x0` outside of domain of `f`"
    end
    y = f.b0
    for (i, x) in enumerate(f.xs[1:end - 1])
        m = f.ms[i]
        xnext = f.xs[i + 1]
        if (x <= x0 < xnext)
            return y + (x0 - x) * m
        else
            y += (xnext - x) * m
        end
    end
    return y
end


support(f::UnivariateLinear) = (f.xs[1], f.xs[end])

function isnonnegative(f::UnivariateLinear)
    b = f.b0
    for (i, x) in enumerate(f.xs[1, end - 1])
        m = f.ms[i]
        xnext = f.xs[i + 1]
        b += (xnext - x) * m
        b < 0. && return false
    end
    return true
end

"""
A univariate point process.

The intensity is given by `λ(t) = λ x h(t)`, where `h` is a valid probability distribution or a piecewise linear function.

- `λ::Float64`: intensity "multiplier".
- `h::Distribution`: intensity shape function (defaults to uniform/constant).
"""
struct PointProcess
    λ::Float64
    h::Union{Distribution,UnivariateLinear,Nothing}
    function PointProcess(λ, h)
        assert_nonnegative(λ, "`λ` must be nonnegative")
        !isnothing(h) && assert_valid_intensity(h)
        return new(λ, h)
    end
end

PointProcess(λ) = PointProcess(λ, nothing)


"""
rand(p::PointProcess, T::Float64)

Construct a random sample from a point process `p` on interval [0, `T`].

First, generate a number of events, `n`. Second, independently sample the time of each event using normalized intensity.
"""
function rand(p::PointProcess, T::Float64)
    if isnothing(p.h)
        n = rand(Poisson(p.λ * T))
        ts = rand(Uniform(0, T), n)
        return sort(ts)
    elseif typeof(p.h) <: Distribution
        s = cdf(p.h, T)
        n = rand(Poisson(p.λ * s))
        ts = rand(truncated(p.h, 0., T), n)
        return sort(ts)
    elseif typeof(p.h) <: Integrable
        @error "Not implemented"
        # s = integral(p.h, [0, T])
        # n = rand(Poisson(p.λ * s))
        # ts = rand()
        # return sort(ts)
    end
end


function intensity(p::PointProcess)
    if isnothing(p.h)
        return t -> p.λ
    elseif typeof(p.h) <: Distribution
        return t -> p.λ * pdf(p.h, t)
    elseif typeof(p.h) <: Integrable
        return t -> p.λ * evaluate(p.h, t)
    end
end

"""

Compute the likelihood of events `ts` given point process `p` and time interval `[0, T]`.
"""
function likelihood(p::PointProcess, ts, T)
    if isnothing(p.h)
        return exp(-p.λ * T) * p.λ^length(ts)
    elseif typeof(p.h) <: Distribution
        return exp(-cdf(p.h, T)) * prod(intensity(p).(ts))
    elseif typeof(p.h) <: Integrable
        return exp(-integral(p.h, T)) * prod(intensity(p).(ts))
    end
end
