using Revise, Pkg
Pkg.activate(".")
using PointProcesses, Gadfly

nchannels = 2;
nlags = 10;
nbasis = 4;
λ0 = 5. * ones(nchannels);
θ = 0.25 * ones(nchannels, nchannels, nbasis);
W = 0.1 * ones(nchannels, nchannels);
Δt = 1.;
αλ = βλ = 1.;
κ = ν = 1.;
γ = 1.;
p = DiscreteNetworkHawkesProcess(λ0, θ, W, nlags, nbasis, Δt, αλ, βλ, κ, ν, γ);

# Plot impulse response
IR = impulse_response(p);
# plot_impulse_responses(IR)

# Generate random data
nsteps = 1000;
events = rand(p, nsteps);
# convolved = PointProcesses.convolve(events, PointProcesses.basis(p));
# λtot = intensity(p, events, convolved);
# plot_intensity(λtot)

# Gibbs sampling
nsamples = 1000;
λ0, θ, W = mcmc(p, events, nsamples);
plot_background_posterior(λ0, burn=500, channels=2)
plot_weight_posterior(W, burn=500, channels=2)
plot_background_chain(λ0, burn=500, channels=2)
plot_weight_chain(W, burn=500, channels=2)


function plot_impulse_responses(IR)
    # IR = impulse_response(p)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel = 1:p.N
        for childchannel = 1:p.N
            ir = IR[parentchannel, childchannel, :]
            plots[parentchannel, childchannel] = plot(x=1:p.D, y=ir)
        end
    end
    gridstack(plots)
end

function plot_intensity(λtot)
    # λtot = intensity(p, events, convolved)
    T, _ = size(λtot)
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        plots[channel] = plot(x=1:T, y=λtot[:, channel], Geom.line)
    end
    vstack(plots)
end

function plot_background_posterior(λ0; burn=0, channels=4)
    # λ0
    plots = Array{Plot,1}(undef, channels)
    for channel in 1:channels
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram(bincount=30), Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_weight_posterior(W; burn=0, channels=4)
    plots = Array{Plot,2}(undef, channels, channels)
    for parentchannel in 1:channels
        for childchannel in 1:channels
            x = [w[parentchannel, childchannel] for w in W[1 + burn:end]]
            plt = plot(x=x, Geom.histogram(bincount=30), Guide.xlabel("W[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

function plot_background_chain(λ0; burn=0, channels=4)
    # λ0
    plots = Array{Plot,1}(undef, channels)
    for channel in 1:channels
        y = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=1:length(y), y=y, Geom.line, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_weight_chain(W; burn=0, channels=4)
    plots = Array{Plot,2}(undef, channels, channels)
    for parentchannel in 1:channels
        for childchannel in 1:channels
            y = [w[parentchannel, childchannel] for w in W[1 + burn:end]]
            plt = plot(x=1:length(y), y=y, Geom.line, Guide.xlabel("W[$parentchannel, $childchannel]"))
            plots[parentchannel, childchannel] = plt
        end
    end
    gridstack(plots)
end

λ0_median = median(cat(λ0..., dims=2), dims=2);
λ0_mean = mean(cat(λ0..., dims=2), dims=2);
W_median = median(cat(W..., dims=3), dims=3);
W_mean = mean(cat(W..., dims=3), dims=3);
θ_median = median(cat(θ..., dims=4), dims=4);
θ_mean = mean(cat(θ..., dims=4), dims=4);
@info "λ0 = $λ0_median"
@info "W = $W_median"
@info "θ = $θ_median"
