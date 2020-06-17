using Revise
using Pkg
Pkg.activate(".")
using PointProcesses

p = HomogeneousProcess(10.);
T = 1.;
ntrials = 1000;
data = [];
for _ in 1:ntrials
    push!(data, rand(p, T));
end
λ = PointProcesses.mle(p, data, T);
@info "λ = $λ"

p = ExponentialProcess(10., 1.);
T = 10.;
ntrials = 1000;
data = [];
for _ in 1:ntrials
    push!(data, rand(p, T));
end
λ = mle(p, data, T);
@info "λ = $λ"

p = LogitNormalProcess(10., 0., 1., 1.);
T = 1.;
ntrials = 1000;
data = [];
for _ in 1:ntrials
    push!(data, rand(p, T));
end
λ = mle(p, data, T);
@info "λ = $λ"

λs = [1., 5., 10., 5., 3., 1., 3., 5., 10., 5., 1.];
p = LinearSplineProcess(0., 1., 10, λs);
T = 10.;
N = 1000.;
data = []
for _ in 1:N
    push!(data, rand(p, T))
end
plot(x=data[1])
ts = 0.:01.:10.;
λ = intensity(p);
ys = [λ(t) for t in ts];
plot(x=ts, y=ys, Geom.Geom.LineGeometry)
@time λ_mle = mle(p, data, T);
p_mle = LinearSplineProcess(0., 1., 10, λ_mle);
λ = intensity(p_mle);
ys_mle = [λ(t) for t in ts];
plot(x=ts, y=ys, Geom.LineGeometry)
@info "λ = $λ_mle"

plot(layer(x=ts, y=ys, Geom.line, Theme(default_color=color("cyan"))),
     layer(x=ts, y=ys_mle, Geom.line), Theme(default_color=color("orange")))
