import Base.rand


"""
An implementation of the network Hawkes model specified in Linderman, 2015.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `A::Array{Int32,2}`: binary matrix indicating network connections.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `μ::Array{Float64,2}`: matrix of impulse-response mean parameters.
- `τ::Array{Float64,2}`: matrix of impulse-response variance parameters.
- `ρ::Array{Float64,2}`: matrix of class connection probabilities.
- `K::Int32`: number of classes in stochastic block model.
- `Δtmax::Float64`: maximum lag parameter.
"""
struct NetworkHawkes
    λ0::Array{Float64,1}
    A::Array{Bool,2}
    W::Array{Float64,2}
    μ::Array{Float64,2}
    τ::Array{Float64,2}
    ρ::Array{Float64,2}
    K::Int32
    Δtmax::Float64
    function NetworkHawkes(λ0, A, W, μ, τ, ρ, K, Δtmax)
        assert_nonnegative(λ0, "Intensity must be non-negative")
        assert_nonnegative(W, "Self-excitement strength must be non-negative")
        assert_nonnegative(τ, "Impulse-response variance must be non-negative")
        assert_probability(ρ, "Connection likelihoods must be valid probabilities.")
        assert_nonnegative(Δtmax, "Maximum lag must be non-negative")
        return new(λ0, A, W, μ, τ, ρ, K, Δtmax)
    end
end


function assert_nonnegative(x, msg)
    if any(x .< 0)
        @error msg
    end
end


function assert_probability(x, msg)
    if any(x .> 1) || any(x .< 0)
        @error msg
    end
end


"""
rand(p::NetworkHawkes, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::NetworkHawkes`: model to sample from.
- `T::Float64`: end of sample period, `[0, T]`.
- `merged::Bool`: return sample as a `(times, nodes)` tuple.
"""
function rand(p::NetworkHawkes, T::Float64; merged = true)

    nchannels = length(p.λ0)

    function generate!(events, parentevent, parentchannel)
        t0 = parentevent
        for childchannel = 1:nchannels
            @info "generating children on channel $childchannel..."
            w = p.W[parentchannel, childchannel]
            μ = p.μ[parentchannel, childchannel]
            τ = p.τ[parentchannel, childchannel]
            h = LogitNormal(μ, τ)
            parent = PointProcess(w, h)
            if t0 + p.Δtmax > T
                childevents = t0 .+ rand(parent, (T - t0) / p.Δtmax) * p.Δtmax
            else
                childevents = t0 .+ rand(parent, 1.) * p.Δtmax
            end
            @info "childevents=$childevents"
            append!(events[childchannel], childevents)
            # generate children of children
            isempty(childevents) && continue
            parentchannel = childchannel
            parentevents = childevents
            for parentevent in parentevents
                @info @sprintf("generating children for event (%.2f, %d)...", parentevent, parentchannel)
                generate!(events, parentevent, parentchannel)
            end
        end
        @info "done."
    end

    # initialize events
    events = Array{Array{Float64,1},1}(undef, nchannels)
    for parentchannel = 1:nchannels
        events[parentchannel] = Array{Float64,1}()
    end
    # generate exogenous events
    for parentchannel = 1:nchannels
        @info "generating exogenous events on channel $parentchannel..."
        parent = PointProcess(p.λ0[parentchannel])
        childevents = rand(parent, T)
        @info "> childevents = $childevents"
        append!(events[parentchannel], childevents)
        # generate endogenous events
        isempty(childevents) && continue
        parentevents = childevents
        for parentevent in parentevents
            @info @sprintf("generating children for event (%.2f, %d)...", parentevent, parentchannel)
            generate!(events, parentevent, parentchannel)
        end
    end
    # sort events
    for parentchannel in 1:nchannels
        sort!(events[parentchannel])
    end
    # format events
    if merged
        times = Array{Float64,1}()
        nodes = Array{Int32,1}()
        for (idx, channelevents) in enumerate(events)
            append!(times, channelevents)
            append!(nodes, idx * ones(length(channelevents)))
        end
        idx = sortperm(times)
        return times[idx], nodes[idx]
    else
        return events
    end
end


"""
impulse_response(p::NetworkHawkes, parentindex, childindex, Δt)

Calculated the impulse-response between `parentindex` and `childindex` after lag `Δt`.
"""
function impulse_response(p::NetworkHawkes, parentindex, childindex, Δt)
    pidx = parentindex
    cidx = childindex
    Δtmax = p.Δtmax
    h = LogitNormal(p.μ[pidx, cidx], p.τ[pidx, cidx])
    return pdf(h, Δt / Δtmax)
end

function impulse_response(p::NetworkHawkes, parentindex, childindex)
    pidx = parentindex
    cidx = childindex
    Δtmax = p.Δtmax
    h = LogitNormal(p.μ[pidx, cidx], p.τ[pidx, cidx])
    ir(Δt) = pdf(h, Δt / Δtmax)
    return ir
end


"""
intensity(p::NetworkHawkes, events, nodes, t0)

Calculate the intensity of `p` at time `t0` given `events` and `nodes`.
"""
function intensity(p::NetworkHawkes, events, nodes, t0)
    nchannels = length(p.λ0)
    # sort
    idx = sortperm(events)
    events = events[idx]
    nodes = nodes[idx]
    # filter
    idx = t0 - p.tmax .< events .< t0
    events = events[idx]
    nodes = nodes[idx]
    # calculate
    λ = zeros(nchannels)
    for i = 1:nchannels
        for (s, c) in zip(events, nodes)
            λ[i] += p.W[c, i] * impulse_response(p, c, i, t0 - s)
        end
    end
    return p.λ0 + λ
end
function intensity(p::NetworkHawkes, events, t0)
    nchannels = length(p.λ0)
    # sort
    sort!.(events)
    # filter
    for i = 1:nchannels
        events[i] = events[i][events[i] .< t0]
    end
    # calculate
    λ = zeros(nchannels)
    for childindex = 1:nchannels
        for parentindex = 1:nchannels
            cidx = childindex
            pidx = parentindex
            λ[i] += sum(p.W[pidx, cidx] .* pdf.(p.H[pidx, cidx], t0 .- events[cidx]))
        end
    end
    return p.λ0 + λ
end


function likelihood(p::NetworkHawkes, events) end
function jointprobability(p::NetworkHawkes, events) end
function stability(p::NetworkHawkes) end
