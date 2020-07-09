using Revise, Distributions, Gadfly
using Pkg
Pkg.activate(".")
using PointProcesses


N = 2
λ0 = 0.1 * ones(N)
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

T = 20000.;
ntrials = 1;
data = [];
for _ in 1:ntrials
    push!(data, rand(p, T));
end

@time λ0_mle, W_mle, θ_mle = mle(p, data, T);
@info λ0_mle
@info W_mle
@info θ_mle
