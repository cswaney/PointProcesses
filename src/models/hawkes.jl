abstract type HawkesProcess end

"""
rand(p::NetworkHawkesProcess, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::NetworkHawkesProcess`: model to sample from.
- `T::Float64`: end of sample period, `[0, T]`.
- `merged::Bool`: return sample as a `(times, nodes)` tuple.
"""
function rand(p::HawkesProcess, T) end

"""
likelihood(p::NetworkHawkesProcess, events, nodes, parents)

Calculate the augmented likelihood of `events`, `nodes`, and `parents` given process `p`.
"""
function likelihood(p::HawkesProcess, events, nodes, T) end

"""
impulse_response(p::NetworkHawkesProcess, parentindex, childindex, Δt)

Calculated the impulse-response between `parentindex` and `childindex` after lag `Δt`.
"""
function impulse_response(p::HawkesProcess) end

"""
intensity(p::NetworkHawkesProcess, events, nodes, t0)

Calculate the intensity of `p` at time `t0` given `events` and `nodes`.
"""
function intensity(p::HawkesProcess, events, nodes, t0) end

"""
stability(p::NetworkHawkesProcess)

Calculate the stability of process `p`. We say that `p` is stable if the retun value is less than one.
"""
function stability(p::HawkesProcess) end


"""
An implementation of the network Hawkes model specified in Linderman, 2015.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `A::Array{Int32,2}`: binary matrix indicating network connections.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `μ::Array{Float64,2}`: matrix of impulse-response mean parameters.
- `τ::Array{Float64,2}`: matrix of impulse-response variance parameters.
- `Δtmax::Float64`: maximum lag parameter.
"""
struct NetworkHawkesProcess <: HawkesProcess
    λ0::Array{Float64,1}
    A::Array{Bool,2}
    W::Array{Float64,2}
    μ::Array{Float64,2}
    τ::Array{Float64,2}
    Δtmax::Float64
    function NetworkHawkesProcess(λ0, A, W, μ, τ, Δtmax)
        assert_nonnegative(λ0, "Intensity must be non-negative")
        assert_nonnegative(W, "Self-excitement strength must be non-negative")
        assert_nonnegative(τ, "Impulse-response variance must be non-negative")
        assert_nonnegative(Δtmax, "Maximum lag must be non-negative")
        return new(λ0, A, W, μ, τ, Δtmax)
    end
end

function rand(p::NetworkHawkesProcess, T::Float64)
    nchannels = length(p.λ0)

    function generate!(events, parentevent, parentchannel)
        t0 = parentevent
        for childchannel = 1:nchannels
            @info "generating children on channel $childchannel..."
            w = p.W[parentchannel, childchannel]
            μ = p.μ[parentchannel, childchannel]
            τ = p.τ[parentchannel, childchannel]
            parent = LogitNormalProcess(w, μ, τ, p.Δtmax)
            childevents = t0 .+ rand(parent, T - t0)
            @info "childevents=$childevents"
            append!(events[childchannel], childevents)
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

    # Create a data structure to hold events
    events = Array{Array{Float64,1},1}(undef, nchannels)
    for parentchannel = 1:nchannels
        events[parentchannel] = Array{Float64,1}()
    end

    # Generate events, starting from exogenous background processes
    for parentchannel = 1:nchannels
        @info "generating exogenous events on channel $parentchannel..."
        parent = HomogeneousProcess(p.λ0[parentchannel])
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

    # Merge events into a single array and create array of marks
    times = Array{Float64,1}()
    nodes = Array{Int32,1}()
    for (idx, channelevents) in enumerate(events)
        append!(times, channelevents)
        append!(nodes, idx * ones(length(channelevents)))
    end
    idx = sortperm(times)
    return times[idx], nodes[idx]
end

function impulse_response(p::NetworkHawkesProcess, parentindex, childindex)
    P = LogitNormalProcess.(p.W, p.μ, p.τ, Δtmax)
    return intensity.(P)
end

function intensity(p::NetworkHawkesProcess, events, nodes, t0)
    nchannels = length(p.λ0)
    idx = t0 - p.Δtmax .< events .< t0
    events = events[idx]
    nodes = nodes[idx]
    λ = zeros(nchannels)
    h = impulse_response(p)
    for childChannel = 1:nchannels
        for (parentTime, parentChannel) in zip(events, nodes)
            λ[i] += h[parentChannel, childChannel](t0 - parentTime)
        end
    end
    return p.λ0 + λ
end

function loglikelihood(p::NetworkHawkesProcess, events, nodes, parents)
    nchannels = length(p.λ0)
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0)
    W = zeros(nchannels)
    for pchannel in nodes
        W .+= p.W[pchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity  TODO: parallelize
    h = impulse_response(p)
    for cidx, ctime, cnode in enumerate(zip(events, nodes))
        λtot = p.λ0[cnode]
        idx = 1
        while events[cidx - idx] > ctime - p.Δtmax
            ptime = events[cidx - idx]
            pchannel = nodes[cidx - idx]
            if p.A[pchannel, cchannel] != 0. && p.W[pchannel, cchannel] != 0.  # NOTE: Not Required
                λtot += impulse_response[pchannel, cchannel](ctime - ptime)
            end
            idx -= 1
        end
        ll += log(λtot)
    end
    return ll
end

stability(p::NetworkHawkesProcess) = maximum(abs(p.A .* p.W))



"""
An implementation of a classic multivariate Hawkes model.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `θ::Array{Float64,2}`: matrix of impulse-response mean parameters.
"""
struct ExponentialHawkesProcess <: HawkesProcess
    λ0
    W
    θ
end

"""
rand(p::ExponentialHawkesProcess, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::ExponentialHawkesProcess`: model to sample from.
- `T::Float64`: end of sample period, `[0, T]`.
"""
function rand(p::ExponentialHawkesProcess, T::Float64)
    nchannels = length(p.λ0)

    function generate!(events, parentevent, parentchannel)
        t0 = parentevent
        for childchannel = 1:nchannels
            @info "generating children on channel $childchannel..."
            w = p.W[parentchannel, childchannel]
            θ = p.θ[parentchannel, childchannel]
            parent = ExponentialProcess(w, θ)
            childevents = t0 .+ rand(parent, T - t0)
            @info "childevents=$childevents"
            append!(events[childchannel], childevents)
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

    # Create a data structure to hold events
    events = Array{Array{Float64,1},1}(undef, nchannels)
    for parentchannel = 1:nchannels
        events[parentchannel] = Array{Float64,1}()
    end

    # Generate events, starting from exogenous background processes
    for parentchannel = 1:nchannels
        @info "generating exogenous events on channel $parentchannel..."
        parent = HomogeneousProcess(p.λ0[parentChannel])
        childevents = rand(parent, T)
        @info "> childevents = $childevents"
        append!(events[parentchannel], childevents)
        isempty(childevents) && continue
        parentevents = childevents
        for parentevent in parentevents
            @info @sprintf("generating children for event (%.2f, %d)...", parentevent, parentchannel)
            generate!(events, parentevent, parentchannel)
        end
    end

    # Merge events into a single array and create array of marks
    times = Array{Float64,1}()
    nodes = Array{Int32,1}()
    for (idx, channelevents) in enumerate(events)
        append!(times, channelevents)
        append!(nodes, idx * ones(length(channelevents)))
    end
    idx = sortperm(times)
    return times[idx], nodes[idx]
end

function impulse_response(p::ExponentialHawkesProcess)
    P = ExponentialProcess.(p.W, p.θ)
    return intensity.(P)
end

function intensity(p::ExponentialHawkesProcess, events, nodes, t0)
    nchannels = length(p.λ0)
    idx = events .< t0
    events = events[idx]
    nodes = nodes[idx]
    λ = zeros(nchannels)
    h = impulse_response(p)
    for childChannel = 1:nchannels
        for (parentTime, parentChannel) in zip(events, nodes)
            λ[i] += h[parentChannel, childChannel](t0 - parentTime)
        end
    end
    return p.λ0 + λ
end

function loglikelihood(p::ExponentialHawkesProcess, events, nodes, T)
    nchannels = length(p.λ0)
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0)
    W = zeros(nchannels)
    for pchannel in nodes
        W .+= p.W[pchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity
    # TODO: parallelize (O(m^2) -> O(m))
    h = impulse_response(p)
    for cidx, ctime, cnode in enumerate(zip(events, nodes))
        λtot = p.λ0[cnode]
        idx = 1
        while idx < cidx
            ptime = events[idx]
            pchannel = nodes[idx]
            λtot += impulse_response[pchannel, cchannel](ctime - ptime)
            idx += 1
        end
        ll += log(λtot)
    end
    return ll
end

stability(p::ExponentialHawkesProcess) = maximum(abs(p.A .* p.W))
