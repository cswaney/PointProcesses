using Distributions, Gadfly

p = HomogeneousProcess(10., 1.)
Ts = 10. * ones(100)
ts = sort(vcat(map(T -> rand(p, T), Ts)...))
plot(ts, Geom.histogram)

# p = LinearProcess(10., 1.)
# Ts = 10. * ones(100)
# ts = sort(vcat(map(T -> rand(p, T), Ts)...))
# plot(ts, Geom.histogram)

p = ExponentialProcess(10., 1.)
Ts = 10. * ones(100)
ts = sort(vcat(map(T -> rand(p, T), Ts)...))
plot(ts, Geom.histogram)

p = LogitNormalProcess(10., 0., 1., 10.)
Ts = 10. * ones(100)
ts = sort(vcat(map(T -> rand(p, T), Ts)...))
plot(ts, Geom.histogram)
