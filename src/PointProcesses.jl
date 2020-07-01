module PointProcesses

using Distributions, Distributed, SharedArrays, Gadfly, Printf, DSP
import Base.rand

include("utils/distributions.jl")
include("utils/assertions.jl")

include("models/Network.jl")
export Network,
       DenseNetwork,
       BernoulliNetwork,
       StochasticBlockNetwork

include("models/Poisson.jl")
include("models/HawkesProcess.jl")
include("models/DiscreteHawkesProcess.jl")
export PoissonProcess,
       HomogeneousProcess,
       MultivariateHomogeneousProcess,
       LinearProcess,
       ExponentialProcess,
       LogitNormalProcess,
       LinearSplineProcess,
       HawkesProcess,
       rand,
       loglikelihood,
       intensity,
       impulse_response,
       stability,
       HawkesProcess,
       StandardHawkesProcess,
       NetworkHawkesProcess,
       DiscreteHawkesProcess

include("inference/mcmc/HawkesProcess.jl")
include("inference/mcmc/DiscreteHawkesProcess.jl")
include("inference/mcmc/Network.jl")
include("inference/vb/DiscreteHawkesProcess.jl")
include("inference/vb/Network.jl")
include("inference/mle/HawkesProcess.jl")
export mcmc, vb, mle

end # module
