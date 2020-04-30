"""
An implementation of a classic multivariate Hawkes model.

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `θ::Array{Float64,2}`: matrix of impulse-response mean parameters.
"""
mutable struct ExponentialHawkesProcess <: HawkesProcess
    # parameters
    λ0::Array{Float64,1}
    W::Array{Float64,2}
    A::Array{Bool,2}
    θ::Array{Float64,2}
    N::Int64
    # hyperparameters
    α0::Float64
    β0::Float64
    κ::Float64
    ν::Array{Float64,2}
    αθ::Float64
    βθ::Float64
    function ExponentialHawkesProcess(λ0, W, θ, N, α0, β0, κ, ν, αθ, βθ)
        return new(λ0, W, ones(N, N), θ, N, α0, β0, κ, ν, αθ, βθ)
    end
end


function generate!(events, parentevent, parentchannel, process::ExponentialHawkesProcess, T)
    t0 = parentevent
    nchannels = process.N
    for childchannel = 1:nchannels
        if process.A[parentchannel, childchannel] == 1
            # @info "generating children on channel $childchannel..."
            w = process.W[parentchannel, childchannel]
            θ = process.θ[parentchannel, childchannel]
            parent = ExponentialProcess(w, θ)
            childevents = t0 .+ rand(parent, T - t0)
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


function loglikelihood(p::ExponentialHawkesProcess, events, nodes, T)
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0) * T
    W = zeros(nchannels)
    for parentchannel in nodes
        W .+= p.W[parentchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity TODO: parallelize
    ir = impulse_response(p)
    for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
        λtot = p.λ0[childchannel]
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = 1
        while parentindex < childindex
            parenttime = events[parentindex]
            parentchannel = nodes[parentindex]
            Δt = childtime - parenttime
            λtot += ir[parentchannel, childchannel](Δt)
            parentindex += 1
        end
        ll += log(λtot)
    end
    return ll
end


function intensity(p::ExponentialHawkesProcess, events, nodes, t0)
    nchannels = p.N
    idx = events .< t0
    events = events[idx]
    nodes = nodes[idx]
    λ = zeros(nchannels)
    ir = impulse_response(p)
    for childchannel = 1:nchannels
        for (parenttime, parentchannel) in zip(events, nodes)
            Δt = t0 - parenttime
            λ[childchannel] += ir[parentchannel, childchannel](t0 - parenttime)
        end
    end
    return p.λ0 + λ
end


function impulse_response(p::ExponentialHawkesProcess)
    P = ExponentialProcess.(p.W, p.θ)
    return intensity.(P)
end
