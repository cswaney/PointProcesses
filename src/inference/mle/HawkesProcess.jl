"""Maximum-Likelihood Estimation

# 1. Homogeneous Process
# 2. Exponential Process
# 3. Logit-Normal Process
# 4. Standard Hawkes Process
# 5. Network Hawkes Process
"""

using Optim
import Optim: minimizer, summary


"""
    mle(process::PoissonProcess)

Find the maximum-likelihood estimate of a Poisson process.

# Arguments
- `events::Array{Array{Float64,1}}`: array of trials.
"""
function mle(process::PoissonProcess) end

function loglikelihood(p::HomogeneousProcess, λ, events, T)
    N = length(events)
    a = -λ * T * N
    b = 0.
    for i = 1:N
        b += length(events[i]) * log(λ)
    end
    return a + b
end

function mle(process::HomogeneousProcess, events, T)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x[1], events, T)
    lower = [0.]
    upper = [Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [λ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function loglikelihood(p::ExponentialProcess, w, θ, events, T)
    N = length(events)
    h = Exponential(1 / θ)
    a = -w * cdf(h, T) * N
    b = 0.
    for i = 1:N
        b += sum(log.(w * pdf.(h, events[i])))
    end
    return a + b
end

function mle(process::ExponentialProcess, events, T)
    w0 = copy(process.w)
    θ0 = copy(process.θ)
    f(x) = -loglikelihood(process, x[1], x[2], events, T)
    lower = [0., 0.]
    upper = [Inf, Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [w0, θ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function loglikelihood(p::LogitNormalProcess, w, μ, τ, events, T)
    N = length(events)
    d = LogitNormal(μ, τ ^ (-1/2))
    a = -w * cdf(d, T / p.Δtmax) * N
    b = 0.
    for i = 1:N
        b += sum(log.(w * pdf.(d, events[i] ./ p.Δtmax)))
    end
    return a + b
end

function mle(process::LogitNormalProcess, events, T)
    w0 = copy(process.w)
    μ0 = copy(process.μ)
    τ0 = copy(process.τ)
    f(x) = -loglikelihood(process, x[1], x[2], x[3], events, T)
    lower = [0., -Inf, 0.]
    upper = [Inf, Inf, Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [w0, μ0, τ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function pack(process::StandardHawkesProcess)
    # return [copy(p.λ0) copy(p.W) copy(p.A) copy(p.θ)]
    return [copy(p.λ0) copy(p.W) copy(p.θ)]
end

function unpack(process::StandardHawkesProcess, arr)
    # λ0 = arr[:, 1]
    # W = arr[:, 2:2 + (N - 1)]
    # A = arr[:, (2 + N):(2 + 2 * N - 1)]
    # θ = arr[:, (2 + 2 * N):(2 + 3 * N - 1)]
    # return λ0, W, A, θ
    λ0 = arr[:, 1]
    W = arr[:, 2:2 + (N - 1)]
    θ = arr[:, (2 + N):(2 + 2 * N - 1)]
    return λ0, W, θ
end

function impulse_response(p::StandardHawkesProcess, W, θ)
    P = ExponentialProcess.(W, θ)
    return intensity.(P)
end

function loglikelihood(p::StandardHawkesProcess, events, nodes, T, λ0, W, θ)
    typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(λ0) * T
    I = zeros(nchannels)
    for parentchannel in nodes
        I .+= W[parentchannel, :]
    end
    ll -= sum(I)
    # Calculate pointwise total intensity
    ir = impulse_response(p, W, θ)
    for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
        λtot = λ0[childchannel]
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = 1
        while parentindex < childindex
            parenttime = events[parentindex]
            parentchannel = nodes[parentindex]
            Δt = childtime - parenttime
            # a = A[parentchannel, childchannel]
            # λtot += a * ir[parentchannel, childchannel](Δt)
            λtot += ir[parentchannel, childchannel](Δt)
            parentindex += 1
        end
        ll += log(λtot)
    end
    return ll
end

function truncated_loglikelihood(p::StandardHawkesProcess, events, nodes, T, λ0, W, θ, Δtmax)
    typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(λ0) * T
    I = zeros(nchannels)
    for parentchannel in nodes
        I .+= W[parentchannel, :]
    end
    ll -= sum(I)
    # Calculate pointwise total intensity
    ir = impulse_response(p, W, θ)
    for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
        λtot = λ0[childchannel]
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = childindex - 1
        while events[parentindex] > childtime - Δtmax
            parenttime = events[parentindex]
            parentchannel = nodes[parentindex]
            Δt = childtime - parenttime
            λtot += ir[parentchannel, childchannel](Δt)
            parentindex -= 1
            parentindex == 0 && break
        end
        ll += log(λtot)
    end
    return ll
end

function mle(p::StandardHawkesProcess, events, nodes, T; Δtmax=Inf)
    x0 = pack(p)

    function f(x)
        λ0, W, θ = unpack(p, x)
        return -truncated_loglikelihood(p, events, nodes, T, λ0, W, θ, Δtmax)
    end

    lower = [zeros(size(p.λ0)) zeros(size(p.W)) zeros(size(p.θ))]
    upper = [Inf * ones(size(p.λ0)) Inf * ones(size(p.W)) Inf * ones(size(p.θ))]
    method = LBFGS()
    start_time = time()
    function status_update(o)
        # println("$(fieldnames(typeof(o)))")
        # println("$(o.metadata)")
        println("iter=$(o.iteration), elapsed=$(time() - start_time)")
        return false
    end
    options = Optim.Options(callback=status_update)
    opt = optimize(f, lower, upper, x0, Fminbox(method), options)
    @info opt
    λ0, W, θ = unpack(p, minimizer(opt))
    return λ0, W, θ
end

function pack(process::NetworkHawkesProcess)
    return [copy(p.λ0) copy(p.W) copy(p.μ) copy(p.τ)]
end

function unpack(process::NetworkHawkesProcess, arr)
    λ0 = arr[:, 1]
    W = arr[:, 2:2 + (N - 1)]
    μ = arr[:, (2 + N):(2 + 2 * N - 1)]
    τ = arr[:, (2 + 2 * N):(2 + 3 * N - 1)]
    return λ0, W, μ, τ
end

function impulse_response(p::NetworkHawkesProcess, W, μ, τ)
    P = LogitNormalProcess.(W, μ, τ, p.Δtmax)
    return intensity.(P)
end

function loglikelihood(p::NetworkHawkesProcess, events, nodes, T, λ0, W, μ, τ)
    typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(λ0) * T
    I = zeros(nchannels)
    for parentchannel in nodes
        I .+= W[parentchannel, :]
    end
    ll -= sum(I)
    # Calculate pointwise total intensity  TODO: parallelize
    ir = impulse_response(p, W, μ, τ)
    for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
        λtot = λ0[childchannel]
        if childindex == 1
            ll += log(λtot)
            continue
        end
        parentindex = childindex - 1
        while events[parentindex] > childtime - p.Δtmax
            parenttime = events[parentindex]
            parentchannel = nodes[parentindex]
            Δt = childtime - parenttime
            λtot += ir[parentchannel, childchannel](Δt)
            parentindex -= 1
            parentindex == 0 && break
        end
        ll += log(λtot)
    end
    return ll
end

function mle(p::NetworkHawkesProcess, events, nodes, T)
    x0 = pack(p)

    function f(x)
        λ0, W, μ, τ = unpack(p, x)
        return -loglikelihood(p, events, nodes, T, λ0, W, μ, τ)
    end

    N = p.N
    lower = [zeros(N) zeros(N,N) -Inf * ones(N,N) zeros(N,N)]
    upper = [Inf * ones(N) Inf * ones(N,N) Inf * ones(N,N) Inf * ones(N,N)]
    method = LBFGS()
    # start_time = time()
    # function status_update(o)
    #     println("iter=$(o.iteration), elapsed=$(time() - start_time)")
    #     return false
    # end
    # options = Optim.Options(callback=status_update)
    # opt = optimize(f, lower, upper, x0, Fminbox(method), options)
    opt = optimize(f, lower, upper, x0, Fminbox(method))
    @info opt
    λ0, W, μ, τ = unpack(p, minimizer(opt))
    return λ0, W, μ, τ
end
