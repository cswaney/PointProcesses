using Revise, Distributions, Gadfly, Distributed
set_default_plot_size(20cm, 20cm)
addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses


N = 4;
K = 2;
λ0 = 0.1 * ones(N);
W = 0.1 * ones(N, N);
A = [1 1 1 1; 1 1 1 1; 1 1 0 0; 1 1 0 0];
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
ρ = 0.5 * ones(K, K);
z = ones(Int64, N);
πz = 1 ./ K * ones(K);
αρ = 1.;
βρ = 1;
γ = 1.;
net = StochasticBlockNetwork(ρ, z, πz, αρ, βρ, γ, N, K);
p = NetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, net);
plot_impulse_respones(p)
T = 50000.;
events, nodes = rand(p, T);
plot_intensity(p, events[1:10], nodes[1:10], 1, events[10])
plot_intensity(p, events[1:10], nodes[1:10], 2, events[10])
nsamples = 2000;
@time λ0, μ, τ, W, A, ρ, z, πz = mcmc(p, (events, nodes, T), nsamples);
burn = 500;
plot_λ0(λ0, burn=burn)
plot_W(W, burn=burn)
plot_A(A, burn=burn)
plot_μ(μ, burn=burn)
plot_τ(τ, burn=burn)
plot_ρ(ρ, burn=burn)
plot_z(z, burn=burn)
plot_πz(πz, burn=burn)


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

function plot_A(A; burn=0)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [a[parentchannel, childchannel] for a in A[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("A[$parentchannel, $childchannel]"))
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

function plot_ρ(ρ; burn=0)
    plots = Array{Plot,2}(undef, p.net.K, p.net.K)
    for parentclass in 1:p.net.K
        for childclass in 1:p.net.K
            x = [ρk[parentclass, childclass] for ρk in ρ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("ρ[$parentclass, $childclass]"))
            plots[parentclass, childclass] = plt
        end
    end
    gridstack(plots)
end

function plot_z(z; burn=0)
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        x = [zn[channel] for zn in z[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("z[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_πz(πz; burn=0)
    plots = Array{Plot,1}(undef, p.net.K)
    for channel in 1:p.net.K
        x = [πzn[channel] for πzn in πz[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("πz[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
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