using Revise, Pkg
Pkg.activate(".")
using PointProcesses
using Distributions, Gadfly


λ0 = ones(2)
W = 0.1 * ones(2, 2)
θ = ones(2, 2)
N = 2
α0 = 1.
β0 = 1.
κ = 1.
ν = ones(2, 2)
αθ = 1.
βθ = 1.
p = ExponentialHawkesProcess(λ0, W, θ, N, α0, β0, κ, ν, αθ, βθ)
T = 1000.;
events, nodes = rand(p, T);
S = 1000;
λ0, W, θ = mcmc(p, (events, nodes, T), S);
plot_posteriors(λ0, W, θ)
ll = loglikelihood(p, events, nodes, T)


function plot_posteriors(λ0, W, θ; burn=0)
    # λ0
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
    # W
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel in 1:p.N
        for childchannel in 1:p.N
            x = [w[parentchannel, childchannel] for w in W[1 + burn:end]]
            plt = plot(x=x, Geom.histogram, Guide.xlabel("W[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
    # θ
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
