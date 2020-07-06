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


function loglikelihood(p::HomogeneousProcess, λ, data, T)
    # N = length(events)
    # a = -λ * T * N
    # b = 0.
    # for i = 1:N
    #     b += length(events[i]) * log(λ)
    # end
    # return a + b
    ll = 0.
    for events in data
        a = -λ * T
        b = length(events) * log(λ)
        ll += a + b
    end
    return ll
end

function prior(p::HomogeneousProcess, λ, θ)
    α, β = θ
    return pdf(Gamma(α, 1 / β), λ)
end

function mle(process::HomogeneousProcess, data, T)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x[1], data, T)
    lower = [0.]
    upper = [Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [λ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function map(process::HomogeneousProcess, data, T, θ)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x[1], data, T) - log(prior(process, x[1], θ))
    lower = [0.]
    upper = [Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [λ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

"""
# Arguments
- `data::Array{Tuple{Array{Float64,1},Array{Int64,1}}, 1}`: array of trial data.
"""
function loglikelihood(p::MultivariateHomogeneousProcess, λ, data, T)
    ll = 0.
    for (events, nodes) in data
        a = -sum(λ) * T
        b = 0.
        for (event, node) in zip(events, nodes)
            b += log(λ[node])
        end
        ll += a + b
    end
    return ll
end

function prior(p::MultivariateHomogeneousProcess, λ, θ)
    """Note: uses the same hyperparameters for all nodes."""
    α, β = θ
    return prod(pdf.(Gamma(α, 1 / β), λ))
end

function mle(process::MultivariateHomogeneousProcess, data, T)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x, data, T)
    lower = zeros(process.N)
    upper = Inf .* ones(process.N)
    method = LBFGS()
    opt = optimize(f, lower, upper, λ0, Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function map(process::MultivariateHomogeneousProcess, data, T, θ)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x, data, T) - log(prior(process, x, θ))
    lower = zeros(process.N)
    upper = Inf .* ones(process.N)
    method = LBFGS()
    opt = optimize(f, lower, upper, λ0, Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function loglikelihood(p::ExponentialProcess, w, θ, data, T)
    # N = length(events)
    # h = Exponential(1 / θ)
    # a = -w * cdf(h, T) * N
    # b = 0.
    # for i = 1:N
    #     b += sum(log.(w * pdf.(h, events[i])))
    # end
    # return a + b
    h = Exponential(1 / θ)
    a = -w * cdf(h, T)
    ll = 0.
    for events in data
        b = sum(log.(w * pdf.(h, events)))
        ll += a + b
    end
    return ll
end

function mle(process::ExponentialProcess, data, T)
    w0 = copy(process.w)
    θ0 = copy(process.θ)
    f(x) = -loglikelihood(process, x[1], x[2], data, T)
    lower = [0., 0.]
    upper = [Inf, Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [w0, θ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end


function loglikelihood(p::LogitNormalProcess, w, μ, τ, data, T)
    # N = length(events)
    # d = LogitNormal(μ, τ ^ (-1/2))
    # a = -w * cdf(d, T / p.Δtmax) * N
    # b = 0.
    # for i = 1:N
    #     b += sum(log.(w * pdf.(d, events[i] ./ p.Δtmax)))
    # end
    # return a + b
    d = LogitNormal(μ, τ ^ (-1/2))
    a = -w * cdf(d, T / p.Δtmax)
    ll = 0.
    for events in data
        b = sum(log.(w * pdf.(d, events ./ p.Δtmax)))
        ll += a + b
    end
    return ll
end

function mle(process::LogitNormalProcess, data, T)
    w0 = copy(process.w)
    μ0 = copy(process.μ)
    τ0 = copy(process.τ)
    f(x) = -loglikelihood(process, x[1], x[2], x[3], data, T)
    lower = [0., -Inf, 0.]
    upper = [Inf, Inf, Inf]
    method = LBFGS()
    opt = optimize(f, lower, upper, [w0, μ0, τ0], Fminbox(method))
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function loglikelihood(p::LinearSplineProcess, λs, data, T)
    ll = 0.
    for events in data
        # integrated intensity
        I = 0.
        for i = 1:p.n
            t0 = p.ts[i]
            t1 = p.ts[i + 1]
            λ0 = λs[i]
            λ1 = λs[i + 1]
            I += p.Δt * (λ0 + 1/2 * (λ1 - λ0))
        end
        a = exp(-I)
        # product of intensity
        b = 1.
        for t in events
            i = min(Int(t ÷ p.Δt + 1), p.n)
            t0 = p.ts[i]
            t1 = p.ts[i + 1]
            λ0 = λs[i]
            λ1 = λs[i + 1]
            b *= λ0 + (λ1 - λ0) * (t - t0) / p.Δt
        end
        ll += log(a * b)
    end
    return ll
end

function mle(process::LinearSplineProcess, data, T)
    x0 = copy(process.λs)
    f(x) = -loglikelihood(process, x, data, T)
    lower = zeros(process.n + 1)
    upper = Inf .* ones(process.n + 1)
    start_time = time()
    function status_update(o)
        println("iter=$(o.iteration), elapsed=$(time() - start_time)")
        return false
    end
    options = Optim.Options(callback=status_update)
    method = LBFGS()
    opt = optimize(f, lower, upper, x0, Fminbox(method), options)
    @info opt
    λmax = minimizer(opt)
    return λmax
end

function pack(p::StandardHawkesProcess)
    # return [copy(p.λ0) copy(p.W) copy(p.A) copy(p.θ)]
    return [copy(p.λ0) copy(p.W) copy(p.θ)]
end

function unpack(process::StandardHawkesProcess, arr)
    # λ0 = arr[:, 1]
    # W = arr[:, 2:2 + (N - 1)]
    # A = arr[:, (2 + N):(2 + 2 * N - 1)]
    # θ = arr[:, (2 + 2 * N):(2 + 3 * N - 1)]
    # return λ0, W, A, θ
    N = process.N
    λ0 = arr[:, 1]
    W = arr[:, 2:2 + (N - 1)]
    θ = arr[:, (2 + N):(2 + 2 * N - 1)]
    return λ0, W, θ
end

function impulse_response(p::StandardHawkesProcess, W, θ)
    P = ExponentialProcess.(W, θ)
    return intensity.(P)
end

"""
# Arguments
`data::Array{Tuple{Array{Float64,1},Array{Float64,1}}},1}`
"""
function loglikelihood(p::StandardHawkesProcess, data, T, λ0, W, θ)

    typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    nchannels = p.N
    ll = 0.
    for (events, nodes) in data
        # Calculate integrated intensity
        ll -= sum(λ0) * T
        I = zeros(nchannels)
        for parentchannel in nodes
            I .+= W[parentchannel, :]
        end
        ll -= sum(I)
        # Calculate intensity product
        R = zeros(p.N, p.N)
        # parentchannel = 0
        parenttimes = zeros(p.N)  # most recent event on each channel
        for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
            λtot = λ0[childchannel]
            # Update recursive sums for child channel
            if parenttimes[childchannel] > 0.
                Δt = childtime - parenttimes[childchannel]
                e = vec(exp.(- Δt ./ θ[childchannel, :]))
                R[childchannel, :] .= e .* (1 .+ R[childchannel, :])
            end
            # Calculate recursive sums from parent channels
            for parentchannel = 1:p.N
                parenttime = parenttimes[parentchannel]
                if parenttime > 0.
                    if parentchannel == childchannel
                        r = R[parentchannel, childchannel]
                        w = W[parentchannel, childchannel]
                        λtot += w * 1 / θ[parentchannel, childchannel] * r
                    else
                        Δt = childtime - parenttime
                        e = exp(- Δt / θ[parentchannel, childchannel])
                        r = 1 + R[parentchannel, childchannel]
                        w = W[parentchannel, childchannel]
                        λtot += w * 1 / θ[parentchannel, childchannel] * e * r
                    end
                end
            end
            # @info "λtot=$λtot"
            ll += log(λtot)
            parenttimes[childchannel] = childtime
        end
    end
    return ll

    # typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    # nchannels = p.N
    # ll = 0.
    # for (events, nodes) in data
    #     # Calculate integrated intensity
    #     ll -= sum(λ0) * T
    #     I = zeros(nchannels)
    #     for parentchannel in nodes
    #         I .+= W[parentchannel, :]
    #     end
    #     ll -= sum(I)
    #     # Calculate intensity product
    #     ir = impulse_response(p, W, θ)
    #     for (childindex, (childtime, childchannel)) in enumerate(zip(events, nodes))
    #         λtot = λ0[childchannel]
    #         if childindex == 1
    #             # @info "λtot=$λtot"
    #             ll += log(λtot)
    #             continue
    #         end
    #         parentindex = 1
    #         while parentindex < childindex
    #             parenttime = events[parentindex]
    #             parentchannel = nodes[parentindex]
    #             Δt = childtime - parenttime
    #             λtot += ir[parentchannel, childchannel](Δt)
    #             parentindex += 1
    #         end
    #         # @info "λtot=$λtot"
    #         ll += log(λtot)
    #     end
    # end
    # return ll
end

function mle(p::StandardHawkesProcess, data, T)
    x0 = pack(p)

    function f(x)
        λ0, W, θ = unpack(p, x)
        # return -loglikelihood(p, data, T, λ0, W, θ)
        return -sum([loglikelihood(p, [d], T, λ0, W, θ) for d in data])
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

function map(p::StandardHawkesProcess, data, T, Δtmax=Inf) end

function pack(p::NetworkHawkesProcess)
    return [copy(p.λ0) copy(p.W) copy(p.μ) copy(p.τ)]
end

function unpack(p::NetworkHawkesProcess, arr)
    λ0 = arr[:, 1]
    W = arr[:, 2:2 + (p.N - 1)]
    μ = arr[:, (2 + p.N):(2 + 2 * p.N - 1)]
    τ = arr[:, (2 + 2 * p.N):(2 + 3 * p.N - 1)]
    return λ0, W, μ, τ
end

function impulse_response(p::NetworkHawkesProcess, W, μ, τ)
    P = LogitNormalProcess.(W, μ, τ, p.Δtmax)
    return intensity.(P)
end

function loglikelihood(p::NetworkHawkesProcess, data, T, λ0, W, μ, τ)
    typeof(p.net) != DenseNetwork && @error "Not implemented for $(typeof(p.net))"
    nchannels = p.N
    ll = 0.
    for (events, nodes) in data
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
    end
    return ll
end

function mle(p::NetworkHawkesProcess, data, T)
    x0 = pack(p)

    function f(x)
        λ0, W, μ, τ = unpack(p, x)
        return -loglikelihood(p, data, T, λ0, W, μ, τ)
    end

    N = p.N
    lower = [zeros(N) zeros(N,N) -Inf * ones(N,N) zeros(N,N)]
    upper = [Inf * ones(N) Inf * ones(N,N) Inf * ones(N,N) Inf * ones(N,N)]
    method = LBFGS()
    start_time = time()
    function status_update(o)
        println("iter=$(o.iteration), elapsed=$(time() - start_time)")
        return false
    end
    options = Optim.Options(callback=status_update)
    opt = optimize(f, lower, upper, x0, Fminbox(method), options)
    @info opt
    λ0, W, μ, τ = unpack(p, minimizer(opt))
    return λ0, W, μ, τ
end

function map(p::NetworkHawkesProcess, data, T) end
