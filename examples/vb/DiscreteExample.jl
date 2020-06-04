import Base.rand
using Distributions, DSP, SpecialFunctions
include("./src/models/Discrete.jl")

nchannels = 2;
nlags = 10;
nbasis = 4;
nsteps = 1000;

# Construct Bernoulli network model
ρ = 0.5;
α = β = 1.;
priors = BernoulliNetworkPriors(α, β);
network = BernoulliNetwork(ρ, priors)

# Construct Discrete Hawkes process
λ0 = 5. * ones(nchannels);
θ = 0.25 * ones(nchannels, nchannels, nbasis);
W = 0.1 * ones(nchannels, nchannels);
A = ones(nchannels, nchannels);
Δt = 1.;
αλ = βλ = 1.;
κ0 = 0.1;
ν0 = 100.;
κ1 = ν1 = 1.;
γ = 1.;
priors = DiscreteProcessPriors(αλ, βλ, γ, κ0, ν0, κ1, ν1);
process = DiscreteProcess(λ0, θ, W, A, Δt, nlags, priors, network);

# Generate sample data
data = rand(process, nsteps);

# Test variational inference
α = ones(nchannels);
β = ones(nchannels);
γ = ones(nchannels, nchannels, nbasis);
κ0 = ones(nchannels, nchannels);
ν0 = ones(nchannels, nchannels);
κ1 = ones(nchannels, nchannels);
ν1 = ones(nchannels, nchannels);
ρ = 0.5 * ones(nchannels, nchannels);
αρ = ones(nchannels, nchannels);
βρ = ones(nchannels, nchannels);
guess = VariationalParameters(α, β, γ, κ0, ν0, κ1, ν1, ρ, αρ, βρ);
# convolved = convolve(data, basis(process));
# @time u = update_parents(process, convolved, α, β, κ1, ν1, κ0, ν0, γ, ρ);
# @time α, β = update_background(process, data, u);
# @time γ = update_impulse_response(process, u);
# @time κ, ν = update_weights(process, data, u);
# @time ρ = update_adjacency_matrix(process, αρ, βρ, ν0, κ0, ν1, κ1);
# @time αρ, βρ = update_network(process, ρ);
@time _α_, _β_, _γ_, _κ0_, _ν0_, _κ1_, _ν1_, _ρ_, _αρ_, _βρ_= svi(process, data, guess, 1000);
