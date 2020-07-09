using Revise, Distributions, Gadfly, Distributed
addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses


N = 12;
λ0 = 0.1 * ones(N);
W = 0.05 * ones(N, N);
A = ones(N, N)
μ = zeros(N, N);
τ = ones(N, N);
Δtmax = 1.;
α0 = 1.;
β0 = 1.;
κ = 1.;
ν = ones(N, N);
μμ = 0.;
κμ = 1.;
ατ = 1.;
βτ = 1.;
net = DenseNetwork(N)
p = NetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, net);
if N == 2
    plot_impulse_respones(p)
end
T = 40000.;
events, nodes = rand(p, T);
if N == 2
    plot_intensity(p, events[1:100], nodes[1:100], 1, events[100])
    plot_intensity(p, events[1:100], nodes[1:100], 2, events[100])
end
nsamples = 1000;
@time λ0, μ, τ, W = mcmc(p, (events, nodes, T), nsamples);
if N == 2
    plot_λ0(λ0, burn=250)
    plot_W(W, burn=250)
    plot_μ(μ, burn=250)
    plot_τ(τ, burn=250)
end
plot(x=1:N, y=median(hcat(λ0[251:end]...), dims=2), Geom.bar)
plot(x=1:N * N, y=reshape(median(cat(W[251:end]..., dims=3), dims=3), N * N), Geom.bar)
plot(x=1:N * N, y=reshape(median(cat(μ[251:end]..., dims=3), dims=3), N * N), Geom.bar)
plot(x=1:N * N, y=reshape(median(cat(τ[251:end]..., dims=3), dims=3), N * N), Geom.bar)

@info median(hcat(λ0...), dims=2)
@info median(cat(W..., dims=3), dims=3)
@info median(cat(μ..., dims=3), dims=3)
@info median(cat(τ..., dims=3), dims=3)


function plot_λ0(λ0; burn=0)
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_W(W; burn=0)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [w[parentchannel, childchannel] for w in W[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("W[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_μ(μ; burn=0)
    # μ
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [t[parentchannel, childchannel] for t in μ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("μ[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_τ(τ; burn=0)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [t[parentchannel, childchannel] for t in τ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("τ[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_impulse_respones(p)
    Δt = 0:0.01:1.
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            ir = impulse_response(p)[parentchannel, childchannel].(Δt)
            plt = plot(x=Δt, y=ir, Geom.line)
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_intensity(p, events, nodes, channel, T)
    t = 0:0.1:T
    λ = [intensity(p, events, nodes, t0)[channel] for t0 in t]
    plt = plot(x=t, y=λ, Geom.line)
    push!(plt, layer(x=events, y=zeros(length(events)), color=[colorant"indianred1"]))
end
