using Distributions, Gadfly

"""Create a synthetic dataset using `NetworkHawkes` model."""
params = Dict(
    :λ0 => 0.1 * ones(2)
    :A => ones(2, 2)
    :W => 0.1 * ones(2, 2)
    :μ => zeros(2, 2)
    :τ => ones(2, 2)
    :ρ => ones(2, 2)
    :K => 2
    :Δtmax => 1.    
)
model = NetworkHawkes(params)
period = 10.
data = rand(model, period)

"""Perform learning and inference using `MCMC` methods."""
priors = # ...
nsamples = 500
mcmc = MCMC(data, priors, nsamples)

"""Compare MCMC results to ground truth parameters. (Note: MCMC is influenced by priors, so posterir distributions are only guaraunteed to center around ground truth given sufficient data)."""
function plot(model::NetworkHawkes, samples)
    λ0 = model.λ0
    λ0_sample = samples.λ0
    # plot histogram with vertical line
    input("Press <enter> to continue")

    A = model.A
    A_sample = samples.A
    # plot histogram with vertical line
    input("Press <enter> to continue")

    W = model.W
    W_sample = samples.W
    # plot histogram with vertical line
    input("Press <enter> to continue")

    μ = model.μ
    μ_sample = samples.μ
    # plot histogram with vertical line
    input("Press <enter> to continue")

    τ = model.τ
    τ_sample = samples.τ
    # plot histogram with vertical line
    input("Press <enter> to continue")

    ρ = model.ρ
    ρ_sample = samples.ρ
    # plot histogram with vertical line
    input("Press <enter> to continue")

    κ = model.κ
    κ_sample = samples.κ
    # plot histogram with vertical line
    input("Press <enter> to continue")
end
