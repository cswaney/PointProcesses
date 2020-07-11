using Revise, Distributions, Gadfly, Distributed
addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses


N = 12
λ0 = 0.1 * ones(N)
W = 0.05 * ones(N, N)
A = ones(N, N)
θ = ones(N, N)
α0 = 1.
β0 = 1.
κ = 1.
ν = ones(N, N)
αθ = 1.
βθ = 1.
net = DenseNetwork(N)
p = StandardHawkesProcess(λ0, W, A, θ, N, α0, β0, κ, ν, αθ, βθ, net)
T = 10000.;
events, nodes = rand(p, T);
nsamples = 1000;
@time λ0, W, θ = mcmc(p, (events, nodes, T), nsamples);
# plot_background(λ0, burn=100)
# plot_weights(W, burn=100)
# plot_impulse_response(θ, burn=100)
plot(x=1:N, y=median(hcat(λ0[251:end]...), dims=2), Geom.bar)
plot(x=1:N * N, y=reshape(median(cat(W[251:end]..., dims=3), dims=3), N * N), Geom.bar)
plot(x=1:N * N, y=reshape(median(cat(θ[251:end]..., dims=3), dims=3), N * N), Geom.bar)


function plot_background(λ0; burn=0)
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_weights(W; burn=0)
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

function plot_impulse_response(θ; burn=0)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [t[parentchannel, childchannel] for t in θ[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("θ[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_impulse_respones(p, Δtmax)
    Δt = 0:0.01:Δtmax
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

function plot_intensity(p, events, nodes, T)
    t = 0:0.1:T
    plots = Array{Plot,1}()
    for channel in 1:p.N
        @info "Plotting intensity on channel $channel..."
        λ = [intensity(p, events, nodes, t0)[channel] for t0 in t]
        plt = plot(x=t, y=λ, Geom.line, Guide.xlabel("t"), Guide.ylabel("λ[$channel]"))
        push!(plt, layer(x=events, y=zeros(length(events)), color=[colorant"indianred1"]))
        push!(plots, plt)
    end
    vstack(plots)
end
