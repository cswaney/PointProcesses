using Distributions, Gadfly

# set parameters
λ0 = 0.1 * ones(2)
A = ones(2, 2)
W = 0.25 * ones(2, 2)
μ = zeros(2, 2)
τ = ones(2, 2)
ρ = ones(2, 2)
K = 2
Δtmax = 1.

# create model
pp = NetworkHawkes(λ0, A, W, μ, τ, ρ, K, Δtmax)

# sample events
T = 10.
events, nodes = rand(pp, T)

# plot events


# plot intensity
# ts = 0.:0.1:T
# λs = [intensity(pp, events, nodes, t0) for t0 in ts]
# ys = [λ[1] for λ in λs]
# plt = plot(x=ts, y=ys, Geom.line)
# ys = [λ[2] for λ in λs]
# plt = plot(x=ts, y=ys, Geom.line)
# ys = [intensity(p, ts, t0) for t0 in ts]
# push!(plt, layer(x=ts, y=ys, color=[colorant"indianred1"]))

# plot impulses
