using Revise, Distributions, Gadfly
using Pkg
Pkg.activate(".")
using PointProcesses


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

T = 2000.;
ntrials = 10;
data = [];
for _ in 1:ntrials
    push!(data, rand(p, T));
end

@time θ_trunc = mle(p, data, T, Δtmax=5.);
@info θ_trunc[1]
@info θ_trunc[2]
@info θ_trunc[3]
