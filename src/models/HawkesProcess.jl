abstract type HawkesProcess end

"""
rand(p::NetworkHawkesProcess, T::Float64)

Sample a random sequence of events from a network Hawkes model.

- `p::NetworkHawkesProcess`: model to sample from.
- `T::Float64`: end of sample period, `[0, T]`.
- `merged::Bool`: return sample as a `(times, nodes)` tuple.
"""
function rand(p::HawkesProcess, T) end

"""
likelihood(p::NetworkHawkesProcess, events, nodes, parents)

Calculate the augmented log-likelihood of `events`, `nodes`, and `parents` given process `p`.
"""
function loglikelihood(p::HawkesProcess, events, nodes, T) end

"""
impulse_response(p::NetworkHawkesProcess, parentindex, childindex, Δt)

Calculated the impulse-response between `parentindex` and `childindex` after lag `Δt`.
"""
function impulse_response(p::HawkesProcess) end

"""
intensity(p::NetworkHawkesProcess, events, nodes, t0)

Calculate the intensity of `p` at time `t0` given `events` and `nodes`.
"""
function intensity(p::HawkesProcess, events, nodes, t0) end

"""
stability(p::NetworkHawkesProcess)

Calculate the stability of process `p`. We say that `p` is stable if the retun value is less than one.
"""
function stability(p::HawkesProcess) end
