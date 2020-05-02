using Revise, Distributions, Gadfly, Distributed
addprocs(8)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses


function construct_process(nchannels)
    λ0 = 0.1 * ones(nchannels)
    μ = zeros(nchannels, nchannels)
    τ = ones(nchannels, nchannels)
    W = 0.1 * rand([0, 1], nchannels, nchannels)
    Δtmax = 1.
    N = nchannels
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(nchannels, nchannels)
    μμ = 0.
    κμ = 1.
    ατ = 1.
    βτ = 1.
    return NetworkHawkesProcess(λ0, μ, τ, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ)
end

T = 20000.;
nsamples = 1000;
nchannels = 10;
p = construct_process(nchannels);
events, nodes = rand(p, T);
λ0, μ, τ, W = mcmc(p, (events, nodes, T), nsamples);
plot_posteriors(λ0, μ, τ, W, A, ρ)
ll = loglikelihood(p, events, nodes, T)

function plot_posteriors(λ0, μ, τ, W, A, ρ, pidx, cidx; burn=0)
    plot(x=[v[cidx] for v in λ0[1 + burn:end]], Geom.histogram(bincount=30))
    @info "median λ0 = $(median([v[cidx] for v in λ0[1 + burn:end]]))"
    plot(x=[v[pidx,cidx] for v in μ[1 + burn:end]], Geom.histogram(bincount=30))
    @info "median μ = $(median([v[pidx,cidx] for v in μ[1 + burn:end]]))"
    plot(x=[v[pidx,cidx] for v in τ[1 + burn:end]], Geom.histogram(bincount=30))
    @info "median τ = $(median([v[pidx,cidx] for v in τ[1 + burn:end]]))"
    plot(x=[v[pidx,cidx] for v in W[1 + burn:end]], Geom.histogram(bincount=30))
    @info "median W = $(median([v[pidx,cidx] for v in W[1 + burn:end]]))"
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
