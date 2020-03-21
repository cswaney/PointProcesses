"""
A univariate Hawkes process.
"""
struct HawkesProcess
    λ::Float64
    h::Distribution
    # tmax::Float64  # TODO
    function HawkesProcess(λ, h)
        # check `λ`
        if λ < 0
            @error "Intensity must be non-negative"
            return nothing
        end
        # check `h`
        supp = support(h)
        if (typeof(supp) != RealInterval)
            @error "Support of `h` must be continuous"
            return nothing
        end
        if (supp.lb < 0)
            @error "Support of `h` must be positive"
            return nothing
        end
        # check `tmax`
        # if tmax < 0
        #     @error "Maximum lag must be non-negative"
        # end
        return new(λ, h)
        new(λ, h, tmax)
    end
end

"""
Sample events from a univariate Hawkes Process, `p`, on interval `[0, T]`.
"""
function rand(p::HawkesProcess, T = 1.)

    function sampleBranch(parent::PointProcess, t0, events)
        # sample children from parent
        childTimes = t0 .+ rand(parent, T - t0)
        @info "t0=$(t0)"
        @info "children=$(childTimes)"
        if length(childTimes) > 0
            # append children to events
            append!(events, childTimes)
            # call sample on each child
            for t0 in childTimes
                childProcess = PointProcess(1., p.h)
                sampleBranch(childProcess, t0, events)
            end
        end
    end

    events = []
    rootProcess = PointProcess(p.λ)
    sampleBranch(rootProcess, 0., events)

    return sort(events)
end


struct MultivariateHawkes
    λ::Array{Float64, 1}  # λ[i] = baseline intensity i
    H::Array{Distribution, 2}  # H[i, j] = impulse response i -> j
    # tmax::Float64

    # inner constructor...
end

function rand(p::MultivariateHawkes, T::Float64 = 1.)

    nchannels = length(p.λ)

    function sampleBranch(parent, t0, events)
        for childChannel = 1:nchannels
            childEvents = t0 .+ rand(parent, T - t0)
            @info "t0=$(t0)"
            @info "c=$(childChannel)"
            @info "children=$(childEvents)"
            append!(events[childChannel], childEvents)
            parentChannel = childChannel
            for parentTime in childEvents
                for childChannel = 1:nchannels
                    parent = PointProcess(1., p.H[parentChannel, childChannel])
                    sampleBranch(parent, parentTime, events)
                end
            end
        end
    end

    events = []
    for parentChannel = 1:nchannels
        parentProcess = PointProcess(p.λ[parentChannel])
        childEvents = rand(parentProcess, T)
        @info "t0=0"
        @info "parentChannel=$(parentChannel)"
        @info "childEvents=$(childEvents)"
        push!(events, childEvents)
    end
    @info events
    for parentChannel = 1:nchannels
        for parentTime in events[parentChannel]
            @info parentTime
            for childChannel = 1:nchannels
                @info childChannel
                parent = PointProcess(1., p.H[parentChannel, childChannel])
                @info parent
                sampleBranch(parent, parentTime, events)
            end
        end
    end

    for parentChannel in 1:nchannels
        sort!(events[parentChannel])
    end

    return events
end


"""
Calculate the intensity of a process given a sequence of events.
"""
function intensity(p::HawkesProcess, ts, t0)
    sort!(ts)  # NOTE: make sure events are ordered1
    ts = ts[ts .< t0]
    if length(ts) == 0
        return p.λ
    else
        return p.λ + sum(pdf.(p.h, t0 .- ts))
    end
end


"""
Calculate the intensity of a process given a sequence of events.

Assume the `ts` is an array of arrays.
"""
function intensity(p::MultivariateHawkes, ts, t0)
    nchannels = length(ts)
    sort!.(ts)  # TODO: is this necessary???
    for i = 1:nchannels
        ts[i] = ts[i][ts[i] .< t0]
    end
    λ = zeros(length(p.λ))
    for i = 1:nchannels
        # compute intensity of channel `i`
        for j = 1:nchannels
            # add contribution from events on channel `j`
            λ[i] += sum(pdf.(p.H[j, i], t0 .- ts[j]))
        end
    end
    return p.λ + λ
end



# TODO
function likelihood(p::PointProcess, ts) end
function likelihood(p::HawkesProcess, ts) end
function likelihood(p::MultivariateHawkes, ts)

    # first integral
    exp(-T * p.λ)

    # second integral
    exp(cdf(p.H[i, j]))  # TODO: check H is defined on (0, T)

end


function joint_probability() end

stability(p::NetworkHawkes) = max(abs.(eig(p.A .* p.W))) < 1
