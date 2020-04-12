"""
A univariate Hawkes process.

- `λ::Float64`: baseline intensity parameter.
- `w::Float64`: self-excitement strength parameter.
- `h::Distribution`: impulse-reponse function.
- `Δtmax::Float64`: maximum lag parameter.
"""
struct UnivariateHawkes
    λ0::Float64
    w::Float64
    h::Distribution
    Δtmax::Float64
    function UnivariateHawkes(λ0, w, h, Δtmax)
        assert_nonnegative(λ0, "`λ0` must be non-negative")
        assert_nonnegative(w, "`w` must be non-negative")
        assert_nonnegative(Δtmax, "`Δtmax` must be non-negative")
        assert_distribution(h)
        return new(λ0, w, h, Δtmax)
    end
end

function UnivariateHawkes(λ0, w, h)
    return UnivariateHawkes(λ0, w, h, Inf)
end

"""
rand(p::UnivariateHawkes, T::Float64)

Sample events from a univariate Hawkes Process, `p`, on the interval `[0, T]`.
"""
function rand(p::UnivariateHawkes, T::Float64)

    function samplebranch(parent::PointProcess, t0, events)
        if t0 + p.Δtmax > T
            childEvents = t0 .+ rand(parent, T - t0)
        else
            childEvents = t0 .+ rand(parent, parent.Δtmax)
        end
        @info "children=$(childEvents)"
        if length(childEvents) > 0
            append!(events, childEvents)
            for t0 in childEvents
                @info @sprintf("generating children for event (%.2f)...", t0)
                samplebranch(PointProcess(p.w, p.h), t0, events)
            end
        end
    end

    events = []
    rootProcess = PointProcess(p.λ0)
    @info "generating root events..."
    rootEvents = rand(rootProcess, T)
    @info "rootEvents=$rootEvents"
    append!(events, rootEvents)
    for t0 in rootEvents
        parent = PointProcess(p.w, p.h)
        @info @sprintf("generating children for event (%.2f)...", t0)
        samplebranch(parent, t0, events)
    end

    return sort(events)
end


"""
intensity(p::UnivariateHawkes, ts, t0)

Calculate the intensity of Hawkes process `p` at time `t0` given events `ts`.
"""
function intensity(p::UnivariateHawkes, ts, t0)
    sort!(ts)
    ts = ts[t0 - p.Δtmax .< ts .< t0]
    if length(ts) == 0
        return p.λ0
    else
        return p.λ0 + sum(p.w .* pdf.(p.h, t0 .- ts))
    end
end

impulse_response(p::UnivariateHawkes, t, t0) = pdf(p.h, t - t0)

"""
likelihood(p::UnivariateHawkes, ts)

Calculate the likelihood of events `ts` given univariate Hawkes process `p`.
"""
function likelihood(p::UnivariateHawkes, ts)
    # 
end


"""
A multivariate Hawkes process.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `W::Array{Float64,2}`: matrix of excitement strength parameters.
- `H::Array{Distribution,2}`: matrix of impulse-reponse functions.
- `Δtmax::Float64`: maximum lag parameter.
"""
struct MultivariateHawkes
    λ0::Array{Float64, 1}  # λ[i] = baseline intensity i
    W::Array{Float64,2}
    H::Array{Distribution{Univariate,Continuous}, 2}  # H[i, j] = impulse response i -> j
    Δtmax::Float64
    function MultivariateHawkes(λ0, W, H, Δtmax)
        if any(λ0 .< 0)
            @error "Intensity must be non-negative"
            return nothing
        end
        if any(W .< 0)
            @error "Self-excitement strength must be non-negative"
            return nothing
        end
        for idx in eachindex(H)
            if support(H[idx]) != RealInterval(0., 1.)
                @error "Support of impulse-response functions must be [0., 1.]"
                return nothing
            end
        end
        if Δtmax < 0
            @error "Maximum lag must be non-negative"
            return nothing
        end
        return new(λ0, W, H, Δtmax)
    end
end

function rand(p::MultivariateHawkes, T::Float64)

    # other way to do this...
    # events = []
    # nodes = []
    # generate events and append times to `events`, node to `nodes`

    nchannels = length(p.λ0)

    function sampleBranch(parent, t0, events)
        for childChannel = 1:nchannels
            if t0 + p.Δtmax > T
                childEvents = t0 .+ rand(parent, (T - t0) / p.Δtmax) * p.Δtmax
            else
                childEvents = t0 .+ rand(parent, 1.) * p.Δtmax
            end
            @info "t0=$(t0)"
            @info "c=$(childChannel)"
            @info "children=$(childEvents)"
            append!(events[childChannel], childEvents)
            parentChannel = childChannel
            for parentTime in childEvents
                for childChannel = 1:nchannels
                    w = p.W[parentChannel, childChannel]
                    h = p.H[parentChannel, childChannel]
                    parent = PointProcess(w, h)
                    sampleBranch(parent, parentTime, events)
                end
            end
        end
    end

    events = []
    nodes = []
    for parentChannel = 1:nchannels
        parentProcess = PointProcess(p.λ0[parentChannel])
        childEvents = rand(parentProcess, T)
        @info "t0=0"
        @info "parentChannel=$(parentChannel)"
        @info "childEvents=$(childEvents)"
        push!(events, childEvents)
    end
    for parentChannel = 1:nchannels
        for parentEvent in events[parentChannel]
            @info "parentEvent=$parentEvent"
            for childChannel = 1:nchannels
                @info childChannel
                w = p.W[parentChannel, childChannel]
                h = p.H[parentChannel, childChannel]
                parent = PointProcess(w, h)
                @info "parent=$parent"
                sampleBranch(parent, parentEvent, events)
            end
        end
    end

    for parentChannel in 1:nchannels
        sort!(events[parentChannel])
    end

    return events
end


"""
Consolidate events from `rand` into a single stream and return a sequence of events times and nodes.
"""
function consolidate(events)
    s = Array{Float64,1}()
    c = Array{Int32,1}()
    for (idx, stream) in enumerate(events)
        append!(s, stream)
        append!(c, idx * ones(length(stream)))
    end
    idx = sortperm(s)
    return s[idx], c[idx]
end



function intensity(p::UnivariateHawkes, ts, xs::Array{Float64,1})
    ys = Array{Float64,1}()
    for x in xs
        y = intensity(p, ts, x)
        push!(ys, y)
    end
    return ys
end

"""
Calculate the intensity at time `t0` of a process given events `ts`.

Assume the `ts` is an array of arrays.
"""
function intensity(p::MultivariateHawkes, ts, t0)
    nchannels = length(p.λ0)
    sort!.(ts)  # TODO: is this necessary???
    for i = 1:nchannels
        ts[i] = ts[i][ts[i] .< t0]
    end
    λ = zeros(nchannels)
    for i = 1:nchannels
        # compute intensity of channel `i`
        for j = 1:nchannels
            # add contribution from events on channel `j`
            λ[i] += sum(W[j, i] .* pdf.(p.H[j, i], t0 .- ts[j]))
        end
    end
    return p.λ0 + λ
end

function intensity(p::MultivariateHawkes, ts, cs, t0)
    nchannels = length(p.λ0)
    # sort
    idx = sortperm(ts)
    ts = ts[idx]
    cs = cs[idx]
    # filter
    idx = t0 - p.Δtmax .< ts .< t0
    ts = ts[idx]
    cs = cs[idx]
    # calculate
    λ = zeros(nchannels)
    for i = 1:nchannels
        for (s, c) in zip(ts, cs)
            λ[i] += W[c, i] * pdf.(p.H[c, i], (t0 - s) / p.Δtmax)
        end
    end
    return p.λ0 + λ
end

impulse_response(p::MultivariateHawkes, i, j, t, t0) = pdf(p.H[i, j], (t - t0) / p.Δtmax)


function likelihood(p::MultivariateHawkes, ts)

    # first integral
    exp(-T * p.λ)

    # second integral
    exp(cdf(p.H[i, j]))  # TODO: check H is defined on (0, T)

end
