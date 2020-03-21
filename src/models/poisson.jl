using Distributions, Random
import Base.rand
import Base.length

# TODO: should `T` be associated with `PointProcess`, or kept as an input to sampling?

"""
A univariate point process.

The intensity is given by `λ(t) = λ x h(t)`, where `h` is a valid probability distribution.
"""
struct PointProcess
    λ::Float64
    h::Distribution
    function PointProcess(λ, h)
        supp = support(h)
        if (typeof(supp) != RealInterval)
            @error "Support of `h` must be continuous"
            return nothing
        end
        if (supp.lb < 0)
            @error "Support of `h` must be positive"
            return nothing
        end
        return new(λ, h)
    end
end

function PointProcess(λ)
    if λ < 0.
        @error "Intensity must be non-negative"
        return nothing
    else
        return PointProcess(λ, Uniform())
    end
end


"""
Construct a random sample from a point process `p` on interval [0, `T`].
"""
function rand(p::PointProcess, T::Float64 = 1.)
    if typeof(p.h) <: Uniform
        # homogeneous
        n = rand(Poisson(p.λ * T))
        ts = rand(Uniform(0, T), n)
        return ts
    else
        # inhomogeneous
        s = cdf(p.h, T)
        n = rand(Poisson(p.λ * s))
        ts = rand(truncated(p.h, 0., T), n)
        return ts
    end
end
