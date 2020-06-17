using Revise, Distributions, Gadfly
using Pkg
Pkg.activate(".")
using PointProcesses


N = 2;
λ0 = 0.1 * ones(N);
W = 0.1 * ones(N, N);
A = ones(N, N)
μ = zeros(N, N);
τ = ones(N, N);
Δtmax = 1.;
α0 = 1.;
β0 = 1.;
κ = 1.;
ν = ones(N, N);
μμ = 0.;
κμ = 1.;
ατ = 1.;
βτ = 1.;
net = DenseNetwork(N)
p = NetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, net);

T = 10000.;
ntrials = 5;
data = []
for _ in 1:ntrials
    push!(data, rand(p, T))
end

@time λ0, W, μ, τ = mle(p, data, T);
@info λ0
@info W
@info μ
@info τ
