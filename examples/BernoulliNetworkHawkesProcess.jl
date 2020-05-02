using Distributions, Gadfly

function construct_process()
    λ0 = 0.5 * ones(2)
    μ = zeros(2, 2)
    τ = ones(2, 2)
    A = [[1., 1.] [1., 0.]]
    W = 0.1 * ones(2, 2)
    ρ = 0.75
    Δtmax = 1.
    N = 2
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(2, 2)
    μμ = 0.
    κμ = 1.
    ατ = 1.
    βτ = 1.
    αρ = 1.
    βρ = 1.
    return BernoulliNetworkHawkesProcess(λ0, μ, τ, A, W, ρ, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, αρ, βρ)
end

T = 10000.;
N = 1000;
p = construct_process();
events, nodes = rand(p, T);
λ0, μ, τ, W, A, ρ = mcmc(p, (events, nodes, T), N);
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
    plt = plot(x=[v for v in ρ[1 + burn:end]], Geom.histogram(bincount=30), Guide.xlabel("ρ"))
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
