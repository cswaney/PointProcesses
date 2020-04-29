abstract type HawkesProcess end


"""
    rand(p::HawkesProcess, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::HawkesProcess`: model to sample from.
- `T::Float64`: end of sample period, `[0, T]`.
- `merged::Bool`: return sample as a `(times, nodes)` tuple.
"""
function rand(p::HawkesProcess, T::Float64)
    nchannels = length(p.λ0)

    # Create a data structure to hold events
    events = Array{Array{Float64,1},1}(undef, nchannels)
    for parentchannel = 1:nchannels
        events[parentchannel] = Array{Float64,1}()
    end

    # Generate events, starting from exogenous background processes
    for parentchannel = 1:nchannels
        # @info "generating exogenous events on channel $parentchannel..."
        parent = HomogeneousProcess(p.λ0[parentchannel])
        childevents = rand(parent, T)
        # @info "> childevents = $childevents"
        append!(events[parentchannel], childevents)
        # generate endogenous events
        isempty(childevents) && continue
        parentevents = childevents
        for parentevent in parentevents
            # @info @sprintf("generating children for event (%.2f, %d)...", parentevent, parentchannel)
            generate!(events, parentevent, parentchannel, p, T)
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


"""
    generate(events, parentevent, parentchannel)

Method called by `rand(p::HawkesProcess, T::Float64)` to generate child events from parent events.

# Arguments
- `events::Array{Float64,1}`: an array holding sampled events.
- `parentevent::Float64`: time of the event generating children.
- `parentchannel::Int64`: channel of the event generating children.
- `process::HawkesProcess`: the root process being sampled.
"""
function generate!(events, parentevent, parentchannel, process::HawkesProcess, T)
    t0 = parentevent
    nchannels = process.N
    for childchannel = 1:nchannels
        if process.A[parentchannel, childchannel] == 1
            # @info "generating children on channel $childchannel..."
            w = process.W[parentchannel, childchannel]
            μ = process.μ[parentchannel, childchannel]
            τ = process.τ[parentchannel, childchannel]
            parent = LogitNormalProcess(w, μ, τ, process.Δtmax)
            childevents = t0 .+ rand(parent, T - t0)
            # TODO: filter childevents < T! (Shouldn't need to based on T - t0 above...)
            # @info "childevents=$childevents"
            append!(events[childchannel], childevents)
            isempty(childevents) && continue
            parentchannel = childchannel
            parentevents = childevents
            for parentevent in parentevents
                # @info @sprintf("generating children for event (%.2f, %d)...", parentevent, parentchannel)
                generate!(events, parentevent, parentchannel, process, T)
            end
        end
    end
    # @info "done."
end


"""
    likelihood(p::HawkesProcess, events, nodes, parents)

Calculate the log-likelihood of `events`, `nodes`, and `parents` given process `p`.

# Notes
The log-likelihood is calculated by marginalizing the augmented likelihood over the latent parent variables. At each event, we calculate the intensity of every potential parent process (including the background process) and add these together to get the "total intensity" at the event.
"""
function loglikelihood(p::HawkesProcess, events, nodes, T)
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0) * T
    W = zeros(nchannels)
    for parentchannel in nodes
        W .+= p.W[parentchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity  TODO: parallelize
    ir = impulse_response(p)
    for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
        λtot = p.λ0[childchannel]
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = childindex - 1
        while events[parentindex] > childtime - p.Δtmax
            parenttime = events[parentindex]
            parentchannel = nodes[parentindex]
            if p.A[parentchannel, childchannel] == 1
                Δt = childtime - parenttime
                λtot += ir[parentchannel, childchannel](Δt)
            end
            parentindex -= 1
            parentindex == 0 && break
        end
        ll += log(λtot)
    end
    return ll
end


"""
    intensity(p::HawkesProcess, events, nodes, t0)

Calculate the intensity of `p` at time `t0` given `events` and `nodes`.
"""
function intensity(p::HawkesProcess, events, nodes, t0)
    nchannels = length(p.λ0)
    idx = t0 - p.Δtmax .< events .< t0
    events = events[idx]
    nodes = nodes[idx]
    λ = zeros(nchannels)
    ir = impulse_response(p)
    for childchannel = 1:nchannels
        for (parenttime, parentchannel) in zip(events, nodes)
            a = p.A[parentchannel, childchannel]
            λ[childchannel] += a * ir[parentchannel, childchannel](t0 - parenttime)
        end
    end
    return p.λ0 + λ
end


"""
    impulse_response(p::HawkesProcess)

Calculate the impulse-response of `p` for all parent-child combinations.
"""
function impulse_response(p::HawkesProcess)
    P = LogitNormalProcess.(p.W, p.μ, p.τ, p.Δtmax)
    return intensity.(P)
end


"""
    stability(p::HawkesProcess)

Calculate the stability of process `p` (`p` is stable if the retun value is less than one).
"""
stability(p::HawkesProcess) = maximum(abs.(p.A .* p.W))
