"""Maximum-Likelihood Estimation

# Issues

- The problem with Optim.jl is that it requires a single vector/matrix argument.
- For homogeneous Poisson processes this is fine: we have a single scalar parameter.
- For multivariate Hawkes processes we have a long list of parameters: θ = {λ0, μ, τ, ...}
- We need a method to pack and unpack model parameters into an array:

    - pack(p, ...) -> Put model parameters into array format
    - unpack(p, arr, ...) -> Extract parameters from array format
"""

using Optim
import Optim: minimizer, summary

UPPER_BDD = 10

function mle(process::PoissonProcess) end

function mle(process::HomogeneousProcess, events, T)
    λ0 = copy(process.λ)
    f(x) = -loglikelihood(process, x, events, T)
    # opt = optimize(f, [λ0], LBFGS())
    opt = optimize(f, 0., 10e6)
    @info opt
    λmax = minimizer(opt)
    # process.λ = λmax
    return λmax
end

function mle(process::ExponentialHawkesProcess, events, nodes, T)
    x0 = pack(process)
    f(x) = -loglikelihood(process, x, events, nodes, T)

    # TODO: Constrain λ0, W, θ > 0.
    # TODO: Print periodic information about optimization progress...
    lower = zeros(size(x0))
    upper = 10. .* ones(size(x0))
    opt = optimize(f, lower, upper, x0, Fminbox(LBFGS()))
    # opt = optimize(f, x0, LBFGS())

    @info opt
    xmle = minimizer(opt)
    λ0, W, θ = unpack(p, xmle)
    # process.λ0 = λ0
    # process.W = W
    # process.θ = θ
    return λ0, W, θ
end


function loglikelihood(process::HomogeneousProcess, λ, events, T)
    a = -λ * T
    b = length(events) * log(λ)
    return a + b
end


function loglikelihood(process::ExponentialHawkesProcess, θ, events, nodes, T)
    λ0, W, θ = unpack(p, θ)
    nchannels = p.N
    ll = 0.
    # Calculate integrated intensity
    ll -= sum(λ0) * T
    W = zeros(nchannels)
    for parentchannel in nodes
        W .+= W[parentchannel, :]
    end
    ll -= sum(W)
    # Calculate pointwise total intensity TODO: parallelize
    ir = impulse_response(p)
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
            λtot += ir[parentchannel, childchannel](Δt)
            parentindex += 1
        end
        ll += log(λtot)
    end
    return ll
end


function pack(process::ExponentialHawkesProcess)
    return [p.λ0 p.W p.θ]  # N x (1 + N + N) = N x (1 + size(W,1) + size(θ,1))
end

function unpack(process::ExponentialHawkesProcess, arr)
    λ0 = arr[:, 1]
    W = arr[:, 2:2 + (N - 1)]
    θ = arr[:, 2 + N:2 + (2N - 1)]
    return λ0, W, θ
end
