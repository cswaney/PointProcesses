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
