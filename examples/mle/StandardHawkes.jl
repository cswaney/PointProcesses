using Revise, Distributions, Gadfly
using Pkg
Pkg.activate(".")
using PointProcesses

# TODO: This is incredible slow... try parallelizing loglikelihood calculation... (1) over individual events... (2) over trials...


N = 2
λ0 = ones(N)
W = 0.1 * ones(N, N)
A = ones(N, N)
θ = ones(N, N)
α0 = 1.
β0 = 1.
κ = 1.
ν = ones(N, N)
αθ = 1.
βθ = 1.
net = DenseNetwork(N)
p = StandardHawkesProcess(λ0, W, A, θ, N, α0, β0, κ, ν, αθ, βθ, net)

T = 200.;
events, nodes = rand(p, T);

# @time θ_mle = mle(p, events, nodes, T);
@time θ_trunc = mle(p, events, nodes, T, Δtmax=5.);
