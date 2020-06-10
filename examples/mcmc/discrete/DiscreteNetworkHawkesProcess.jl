using Revise, Distributions, Gadfly, Distributed
addprocs(4)
@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere using PointProcesses


N = 2;
B = 4;
L = 10;
λ0 = 5. * ones(N);
θ = 0.25 * ones(N, N, B);
W = 0.1 * ones(N, N);
A = ones(N, N);
Δt = 1.;
αλ = βλ = 1.;
γ = 1.;
κ0 = ν0 = 1.;
κ1 = ν1 = 1.;  # not used
net = DenseNetwork(N);
p = DiscreteHawkesProcess(λ0, θ, W, A, Δt, N, B, L, αλ, βλ, γ, κ0, ν0, κ1, ν1, net);

# Plot impulse response
IR = impulse_response(p);
plot_impulse_responses(IR)

# Generate random data
nsteps = 20000;
events = rand(p, nsteps);
convolved = PointProcesses.convolve(events, PointProcesses.basis(p));
λtot = intensity(p, events, convolved);
plot_intensity(λtot)

# Gibbs sampling
nsamples = 2000;
λ0, θ, W = mcmc(p, events, nsamples);
burn = 500;
plot_λ0(λ0, burn=500, channels=2)
plot_W(W, burn=500, channels=2)
plot_λ0_chain(λ0, burn=500, channels=2)
plot_W_chain(W, burn=500, channels=2)


function plot_impulse_responses(IR)
    plots = Array{Plot,2}(undef, p.N, p.N)
    for parentchannel = 1:p.N
        for childchannel = 1:p.N
            ir = IR[parentchannel, childchannel, :]
            plots[parentchannel, childchannel] = plot(x=1:p.L, y=ir)
        end
    end
    gridstack(plots)
end

function plot_intensity(λtot)
    T, _ = size(λtot)
    plots = Array{Plot,1}(undef, p.N)
    for channel in 1:p.N
        plots[channel] = plot(x=1:T, y=λtot[:, channel], Geom.line)
    end
    vstack(plots)
end

function plot_λ0(λ0; burn=0, channels=4)
    plots = Array{Plot,1}(undef, channels)
    for channel in 1:channels
        x = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=x, Geom.histogram(bincount=30), Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_W(W; burn=0, channels=4)
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

function plot_λ0_chain(λ0; burn=0, channels=4)
    plots = Array{Plot,1}(undef, channels)
    for channel in 1:channels
        y = [λ[channel] for λ in λ0[1 + burn:end]]
        plt = plot(x=1:length(y), y=y, Geom.line, Guide.xlabel("λ0[$channel]"))
        plots[channel] = plt
    end
    vstack(plots)
end

function plot_W_chain(W; burn=0, channels=4)
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
@info "Posterior Medians:"
@info "λ0 = $λ0_median"
@info "W = $W_median"
@info "θ = $θ_median"
@info "Posterior Means:"
@info "λ0 = $λ0_mean"
@info "W = $W_mean"
@info "θ = $θ_mean"
