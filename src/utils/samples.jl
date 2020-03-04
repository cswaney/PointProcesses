import Base.rand

# TODO: create method to sample from inhomogeneous point process. **NOTE**: a trick is to require the impulse response to come from a family that is a valid distribution (or related to one) such that it integrates to 1.0 (or another known value).

"""
A univariate homogeneous point process.
"""
struct PointProcess
    λ
end

"""
Construct a random sample from homogeneous point process `p` on interval [0, `T`].
"""
function rand(p::PointProcess, T = 1.)
    n = rand(Poisson(p.λ * T))
    ts = rand(Uniform(), n)
    return ts
end

"""
A univariate Hawkes process.
"""
struct HawkesProcess
    λ
    h  # impulse response
    tmax  # maxiumum response
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
