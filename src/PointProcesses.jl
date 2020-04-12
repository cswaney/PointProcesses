module PointProcesses

using Distributions, Distributed, Gadfly, Printf

include("utils/normal_gamma.jl")
export NormalGamma

include("utils/assert.jl")
export assert_nonnegative, assert_distribution, assert_probability

include("models/poisson.jl")
export PoissonProcess

include("models/hawkes.jl")
export UnivariateHawkes, MultivariateHawkes, impulse_response, intensity, likelihood

include("models/network_hawkes.jl")
export NetworkHawkes, impulse_response, intensity

end # module
