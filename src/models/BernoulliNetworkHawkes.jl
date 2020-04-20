"""
An implementation of the network Hawkes model specified in Linderman, 2015.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `A::Array{Int32,2}`: binary matrix indicating network connections.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `μ::Array{Float64,2}`: matrix of impulse-response mean parameters.
- `τ::Array{Float64,2}`: matrix of impulse-response variance parameters.
- `Δtmax::Float64`: maximum lag parameter.
"""
struct BernoulliNetworkHawkesProcess <: HawkesProcess
    # parameters
    λ0::Array{Float64,1}
    μ::Array{Float64,2}
    τ::Array{Float64,2}
    A::Array{Bool,2}
    W::Array{Float64,2}
    Δtmax::Float64
    N::Int32
    # hyperparameters
    α0::Float64
    β0::Float64
    κ::Float64
    ν::Array{Float64,2}
    μμ::Float64
    κμ::Float64
    ατ::Float64
    βτ::Float64
    αρ::Float64
    βρ::Float64
    function BernoulliNetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, αρ, βρ)
        assert_nonnegative(λ0, "Intensity must be non-negative")
        assert_nonnegative(W, "Self-excitement strength must be non-negative")
        assert_nonnegative(τ, "Impulse-response variance must be non-negative")
        assert_nonnegative(Δtmax, "Maximum lag must be non-negative")
        # TODO: finish assertions
        return new(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μ_μ, κ_μ, α_τ, β_τ, α_ρ, β_ρ)
    end
end

function rand(p::BernoulliNetworkHawkesProcess, T::Float64)
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

function loglikelihood(p::BernoulliNetworkHawkesProcess, events, nodes, parents)
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

function intensity(p::BernoulliNetworkHawkesProcess, events, nodes, t0)
    nchannels = length(p.λ0)
    idx = t0 - p.Δtmax .< events .< t0
    events = events[idx]
    nodes = nodes[idx]
    λ = zeros(nchannels)
    h = impulse_response(p)
    for childchannel = 1:nchannels
        for (parenttime, parentchannel) in zip(events, nodes)
            a = p.A[parentchannel, childchannel]
            λ[i] +=  a * h[parentchannel, childchannel](t0 - parenttime)
        end
    end
    return p.λ0 + λ
end

function impulse_response(p::BernoulliNetworkHawkesProcess, parentindex, childindex)
    P = LogitNormalProcess.(p.W, p.μ, p.τ, Δtmax)
    return intensity.(P)
end
