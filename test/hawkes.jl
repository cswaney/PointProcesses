using Distributions, Gadfly

# Hawkes
λ0 = 1.
w = 1.
h = Exponential()
p = UnivariateHawkes(λ0, w, h)
T = 5.
ts = rand(p, T)

# Plotting
xs = 0.0:0.01:T
λs = [intensity(p, ts, t0) for t0 in xs]
plt = plot(x=xs, y=λs, Geom.line)
ys = [intensity(p, ts, t0) for t0 in ts]
push!(plt, layer(x=ts, y=ys, color=[colorant"indianred1"]))
