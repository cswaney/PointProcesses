abstract type HawkesProcess end


"""
An implementation of the network Hawkes model specified in (Linderman, 2015).

- `λ0::Array{Float64,1}`: baseline intensity parameter.
- `A::Array{Int32,2}`: binary matrix indicating network connections.
- `W::Array{Float64,2}`: matrix of network connection strength parameters.
- `μ::Array{Float64,2}`: matrix of impulse-response mean parameters.
- `τ::Array{Float64,2}`: matrix of impulse-response variance parameters.
- `Δtmax::Float64`: maximum lag parameter.
"""
mutable struct NetworkHawkesProcess <: HawkesProcess
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
    # network
    net::Network
end

function NetworkHawkesProcess(N, Δtmax, net)
    λ0 = 0.1 * ones(N)
    W = 0.05 * ones(N, N)
    A = ones(N, N)
    μ = zeros(N, N)
    τ = ones(N, N)
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(N, N)
    μμ = 0.
    κμ = 1.
    ατ = 1.
    βτ = 1.
    p = NetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, net)
    return p
end


mutable struct StandardHawkesProcess <: HawkesProcess
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
    # network
    net::Network
end

function StandardHawkesProcess(N)
    λ0 = ones(N)
    W = 0.1 * ones(N, N)
    A = ones(N, N)
    θ = ones(N, N)
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(N, N)
    αθ = 1.
    βθ = 1.
    net = DenseNetwork(N)
    return StandardHawkesProcess(λ0, W, A, θ, N, α0, β0, κ, ν, αθ, βθ, net)
end

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
function generate!(events, parentevent, parentchannel, process::NetworkHawkesProcess, T)
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

function generate!(events, parentevent, parentchannel, process::StandardHawkesProcess, T)
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


"""
    likelihood(p::HawkesProcess, events, nodes, parents)

Calculate the log-likelihood of `events`, `nodes`, and `parents` given process `p`.

# Notes
The log-likelihood is calculated by marginalizing the augmented likelihood over the latent parent variables. At each event, we calculate the intensity of every potential parent process (including the background process) and add these together to get the "total intensity" at the event.
"""
function loglikelihood(p::NetworkHawkesProcess, events, nodes, T)
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

function loglikelihood(p::StandardHawkesProcess, events, nodes, T)
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

function augmented_loglikelihood(p::NetworkHawkesProcess, events, nodes, parents, T)
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0) * T
    W = zeros(nchannels)
    for parentchannel in nodes
        W .+= p.W[parentchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity
    ir = impulse_response(p)
    for (childindex, (childtime, childchannel, parentindex)) in enumerate(zip(events, nodes, parents))
        if parentindex > 0  # spawned event
            parentchannel = nodes[parentindex]
            parenttime = events[parentindex]
            Δt = childtime - parenttime
            if p.A[parentchannel, childchannel] == 1
                ll += log(ir[parentchannel, childchannel](Δt))
            end
        else  # background event
            ll += log(p.λ0[childchannel])
        end
    end
    return ll
end

function augmented_loglikelihood(p::StandardHawkesProcess, events, nodes, parents, T)
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(p.λ0) * T
    W = zeros(nchannels)
    for parentchannel in nodes
        W .+= p.W[parentchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity
    ir = impulse_response(p)
    for (childindex, (childtime, childchannel, parentindex)) in enumerate(zip(events, nodes, parents))
        if parentindex > 0  # spawned event
            parentchannel = nodes[parentindex]
            parenttime = events[parentindex]
            Δt = childtime - parenttime
            ll += log(ir[parentchannel, childchannel](Δt))
        else  # background event
            ll += log(p.λ0[childchannel])
        end
    end
    return ll
end

function predictive_loglikelihood(p::NetworkHawkesProcess, sample, events, nodes, T)
    pcopy = deepcopy(p)
    parents, λ0, μ, τ, A, W = sample
    L = length(parents)
    ll = 0.
    for i = 1:L
        pcopy.λ0 = λ0[i]
        pcopy.μ = μ[i]
        pcopy.τ = τ[i]
        pcopy.A = A[i]
        pcopy.W = W[i]
        ll += augmented_loglikelihood(p, events, nodes, parents[i], T)
    end
    return ll / L
end

function predictive_loglikelihood(p::StandardHawkesProcess, sample, events, nodes, T)
    pcopy = deepcopy(p)
    parents, λ, θ, A, W = sample
    L = length(parents)
    ll = 0.
    for i = 1:L
        pcopy.λ = λ[i]
        pcopy.θ = θ[i]
        pcopy.A = A[i]
        pcopy.W = W[i]
        ll += augmented_loglikelihood(p, events, nodes, parents[i], T)
    end
    return ll / L
    return ll
end


"""
    intensity(p::HawkesProcess, events, nodes, t0)

Calculate the intensity of `p` at time `t0` given `events` and `nodes`.
"""
function intensity(p::NetworkHawkesProcess, events, nodes, t0)
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

function intensity(p::StandardHawkesProcess, events, nodes, t0)
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


"""
    impulse_response(p::HawkesProcess)

Calculate the impulse-response of `p` for all parent-child combinations.
"""
function impulse_response(p::NetworkHawkesProcess)
    P = LogitNormalProcess.(p.W, p.μ, p.τ, p.Δtmax)
    return intensity.(P)
end

function impulse_response(p::StandardHawkesProcess)
    P = ExponentialProcess.(p.W, p.θ)
    return intensity.(P)
end


"""
    stability(p::HawkesProcess)

Calculate the stability of process `p` (`p` is stable if the retun value is less than one).
"""
stability(p::HawkesProcess) = maximum(abs.(p.A .* p.W))
