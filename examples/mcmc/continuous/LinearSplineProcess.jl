using Revise, Pkg
Pkg.activate(".")
using PointProcesses

λs = [1., 5., 10., 5., 3., 1., 3., 5., 10., 5., 1.];
p = LinearSplineProcess(0., 1., 10, λs);
T = 10.;
N = 100.;
data = []
for _ in 1:N
    push!(data, rand(p, T))
end
# events = rand(p, T);
ts = 0.:01.:10.;
λ = intensity(p);
ys = [λ(t) for t in ts];
plot(x=ts, y=ys, Geom.LineGeometry)

λ_mcmc = mcmc(p, data, 1000);
# λ_mcmc = mcmc(p, events, 1000);


λ_mcmc = hcat(λ_mcmc...)
p_mcmc = LinearSplineProcess(0., 1., 10, vec(mean(λ_mcmc, dims=2)));
λ = intensity(p_mcmc);
ys_mcmc = [λ(t) for t in ts];
# plot(x=ts, y=ys_mcmc, Geom.LineGeometry)
@info "λ = $(mean(λ_mcmc))"

plot(layer(x=ts, y=ys, Geom.line, Theme(default_color=color("cyan"))),
     layer(x=ts, y=ys_mcmc, Geom.line), Theme(default_color=color("orange")))

plot(x=λ_mcmc[1, :], Geom.histogram)
#
