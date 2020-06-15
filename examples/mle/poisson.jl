using Revise
using Pkg
Pkg.activate(".")
using PointProcesses

p = HomogeneousProcess(10.);
T = 1.;
M = 100;
events = [rand(p, T) for _ in 1:M];
λ = mle(p, events, T);
@info "λ = $λ"

p = ExponentialProcess(10., 1.);
T = 1.;
M = 100;
events = [rand(p, T) for _ in 1:M];
λ = mle(p, events, T);
@info "λ = $λ"

p = LogitNormalProcess(10., 0., 1., 1.);
T = 1.;
M = 100;
events = [rand(p, T) for _ in 1:M];
λ = mle(p, events, T);
@info "λ = $λ"

λs = [1., 5., 10., 5., 3., 1., 3., 5., 10., 5., 1.];
p = PointProcesses.LinearSplineProcess(0., 1., 10, λs);
T = 10.;
events = PointProcesses.rand(p, T);
plot(x=events)
ts = 0.:01.:10.;
λ = PointProcesses.intensity(p);
ys = [λ(t) for t in ts];
plot(x=ts, y=ys, Geom.Geom.LineGeometry)
λ_mle = PointProcesses.mle(p, events, T);
p_mle = PointProcesses.LinearSplineProcess(0., 1., 10, λ_mle);
λ = PointProcesses.intensity(p_mle);
ys = [λ(t) for t in ts];
plot(x=ts, y=ys, Geom.LineGeometry)
