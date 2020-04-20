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
