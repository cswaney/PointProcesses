using Revise, Distributions, Gadfly, Distributed
addprocs(8)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses

T = 20000.;
nsamples = 1000;
nchannels = 10;

λ0 = 0.1 * ones(nchannels);
μ = zeros(nchannels, nchannels);
τ = ones(nchannels, nchannels);
W = 0.1 * rand([0, 1], nchannels, nchannels);
Δtmax = 1.;
N = nchannels;
α0 = 1.;
β0 = 1.;
κ = 1.;
ν = ones(nchannels, nchannels);
μμ = 0.;
κμ = 1.;
ατ = 1.;
βτ = 1.;
p = NetworkHawkesProcess(λ0, μ, τ, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ);
events, nodes = rand(p, T);
@time λ0, μ, τ, W = mcmc(p, (events, nodes, T), nsamples);
plot_posteriors(λ0, μ, τ, W, A, ρ)
ll = loglikelihood(p, events, nodes, T)

function plot_posteriors(λ0, W, μ, τ; burn=0, maxchannel=4)
    # λ0
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:maxchannel
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
    # W
    plots = Array{Plot,2}(undef, maxchannel, maxchannel)
    for parentchannel in 1:maxchannel
        for childchannel in 1:maxchannel
            x = [w[parentchannel, childchannel] for w in W[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("W[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
    # μ
    plots = Array{Plot,2}(undef, maxchannel, maxchannel)
    for parentchannel in 1:maxchannel
        for childchannel in 1:maxchannel
            x = [t[parentchannel, childchannel] for t in μ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("μ[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
    # τ
    plots = Array{Plot,2}(undef, maxchannel, maxchannel)
    for parentchannel in 1:maxchannel
        for childchannel in 1:maxchannel
            x = [t[parentchannel, childchannel] for t in τ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("τ[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_impulse_respones(p, parentchannel, childchannel)
    Δt = 0:0.01:1.
    ir = impulse_response(p)[parentchannel, childchannel].(Δt)
    plot(x=Δt, y=ir, Geom.line)
end

function plot_intensity(p, events, nodes, channel, T)
    t = 0:0.1:T
    λ = [intensity(p, events, nodes, t0)[channel] for t0 in t]
    plt = plot(x=t, y=λ, Geom.line)
    push!(plt, layer(x=events, y=zeros(length(events)), color=[colorant"indianred1"]))
end
