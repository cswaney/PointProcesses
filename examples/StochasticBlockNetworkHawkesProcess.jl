using Revise, Pkg
Pkg.activate(".")
using PointProcesses
using Distributions, Gadfly

# 1. Store/print initial parameters
# 2. Plot all parameters; include original value indicators
# 3. Suppress @info
# 4. Time Gibbs sampling

function construct_process(N)
    K = 3
    λ0 = 0.5 * ones(N)
    μ = zeros(N, N)
    τ = ones(N, N)
    W = 0.1 * ones(N, N)
    π = 1 / K * ones(K)
    z = sort(PointProcesses.dense(rand(PointProcesses.Discrete(π), N)))
    ρ = [[1., 1., 1.] [0.5, 0.5, 0.5] [0.1, 0.1, 0.1]]
    A = rand.(Bernoulli.(ρ[z, z]))
    Δtmax = 1.
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(N, N)
    μμ = 0.
    κμ = 1.
    ατ = 1.
    βτ = 1.
    αρ = 1.
    βρ = 1.
    γ = 1.
    return StochasticBlockNetworkHawkesProcess(λ0, μ, τ, A, W, ρ, z, π, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, αρ, βρ, γ, K)
end

T = 5000.;
N = 1000;
p = construct_process(6);
events, nodes = rand(p, T);
λ0, μ, τ, W, A, ρ, z, π = mcmc(p, (events, nodes, T), N);
#
plot_posteriors(λ0, μ, τ, W, A, ρ)
ll = loglikelihood(p, events, nodes, T)

function plot_posteriors(λ0, μ, τ, W, A, ρ; burn=0)
    # λ0
    plt1 = plot(x=[v[1] for v in λ0[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("λ0[1]"));
    plt2 = plot(x=[v[2] for v in λ0[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("λ0[2]"));
    gridstack([plt1 plt2])
    # μ
    plt11 = plot(x=[v[1,1] for v in μ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("μ[11]"));
    plt12 = plot(x=[v[1,2] for v in μ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("μ[12]"));
    plt21 = plot(x=[v[2,1] for v in μ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("μ[21]"));
    plt22 = plot(x=[v[2,2] for v in μ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("μ[22]"));
    gridstack([plt11 plt12; plt21 plt22])
    # τ
    plt11 = plot(x=[v[1,1] for v in τ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("τ[11]"));
    plt12 = plot(x=[v[1,2] for v in τ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("τ[12]"));
    plt21 = plot(x=[v[2,1] for v in τ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("τ[21]"));
    plt22 = plot(x=[v[2,2] for v in τ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("τ[22]"));
    gridstack([plt11 plt12; plt21 plt22])
    # W
    plt11 = plot(x=[v[1,1] for v in W[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("W[11]"));
    plt12 = plot(x=[v[1,2] for v in W[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("W[12]"));
    plt21 = plot(x=[v[2,1] for v in W[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("W[21]"));
    plt22 = plot(x=[v[2,2] for v in W[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("W[22]"));
    gridstack([plt11 plt12; plt21 plt22])
    # A
    plt11 = plot(x=[v[1,1] for v in A[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("A[11]"));
    plt12 = plot(x=[v[1,2] for v in A[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("A[12]"));
    plt21 = plot(x=[v[2,1] for v in A[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("A[21]"));
    plt22 = plot(x=[v[2,2] for v in A[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("A[22]"));
    gridstack([plt11 plt12; plt21 plt22])
    # ρ
    plt11 = plot(x=[v[1,1] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[11]"));
    plt12 = plot(x=[v[1,2] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[12]"));
    plt21 = plot(x=[v[2,1] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[21]"));
    plt22 = plot(x=[v[2,2] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[22]"));
    gridstack([plt11 plt12; plt21 plt22])
    # z
    plt11 = plot(x=[v[1,1] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[11]"));
    plt12 = plot(x=[v[1,2] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[12]"));
    plt21 = plot(x=[v[2,1] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[21]"));
    plt22 = plot(x=[v[2,2] for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ[22]"));
    gridstack([plt11 plt12; plt21 plt22])
end

function plot_impulse_respones(parentchannel, childchannel)
    Δt = 0:0.01:1.
    ir = impulse_response(p)[parentchannel, childchannel].(Δt)
    plot(x=Δt, y=ir, Geom.line)
end

function plot_intensity(channel)
    t = 0:0.1:T
    λ = [intensity(p, events, nodes, t0)[channel] for t0 in t]
    plt = plot(x=t, y=λ, Geom.line)
    push!(plt, layer(x=events, y=zeros(length(events)), color=[colorant"indianred1"]))
end
