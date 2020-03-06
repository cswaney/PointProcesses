import Base.rand

# TODO: consolidate homogeneous and inhomogeneous point processes!

"""
A univariate homogeneous point process.
"""
struct PointProcess
    λ
end


"""
Construct a random sample from homogeneous point process `p` on interval [0, `T`].
"""
function rand(p::PointProcess, T::Float64 = 1.)
    n = rand(Poisson(p.λ * T))
    ts = rand(Uniform(0, T), n)
    return ts
end

# TODO: match Distributions API
function rand(p::PointProcess, T::Float64 = 1., n::Int32 = 1)
    if n == 1
        n = rand(Poisson(p.λ * T))
        ts = rand(Uniform(), n)
        return ts
    elseif n > 1
        ts = []
        for i = 1:n
            n = rand(Poisson(p.λ * T))
            ts.append(rand(Uniform(), n))
        end
        return ts
    else
        @error "Sample size `n` must be positive"
    end
end


"""
A univariate inhomogeneous point process. The intensity is given by `λ x h(t)`
"""
struct InhomogeneousPointProcess
    h::Distribution
    λ
    function InhomogeneousPointProcess(h, λ)
        supp = support(h)
        if (typeof(supp) != RealInterval)
            @error "Support of `h` must be continuous"
            return nothing
        end
        if (supp.lb < 0)
            @error "Support of `h` must be positive"
            return nothing
        end
        return new(h, λ)
    end
end


"""
Construct a random sample from inhomogeneous point process `p` on interval [0, `T`].
"""
function rand(p::InhomogeneousPointProcess, T::Float64 = 1.)
    s = cdf(p.h, T)
    n = rand(Poisson(p.λ * s))
    ts = rand(truncated(p.h, 0., T), n)
    return ts
end

# TODO: match Distributions API
function rand(p::InhomogeneousPointProcess, n::Int32 = 1)
    if n == 1
        ts = rand(p.h, rand(Poisson(p.λ)))
        return ts
    elseif n > 1
        ts = []
        for i = 1:n
            ts.append(rand(p.h, rand(Poisson(p.λ))))
        end
        return ts
    else
        @error "Sample size `n` must be positive"
    end
end


"""
A univariate Hawkes process.
"""
struct HawkesProcess
    λ
    h::Distribution  # impulse response
    tmax  # maxiumum response
    function HawkesProcess(λ, h, tmax)

        # check that λ, h, and tmax are kosher...

        new(λ, h, tmax)
    end
end

"""
Sample events from a univariate Hawkes Process, `p`, on interval `[0, T]`.
"""
function rand(p::HawkesProcess, T = 1.)

    function sampleBranch(parent::PointProcess, t0, events)
        # sample children from parent
        children = rand(parent, T - t0)
        if length(children) > 0
            # append children to events
            append!(events, children)
            # call sample on each child
            for child in children
                sampleBranch(PointProcess(p.h), child, events)
            end
        end
    end

    events = []
    sampleBranch(PointProcess(p.λ), 0., events)

    return events
end
