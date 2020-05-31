module PointProcesses

using Distributions, Distributed, SharedArrays, Gadfly, Printf
import Base.rand

include("utils/distributions.jl")
include("utils/assertions.jl")

include("models/Poisson.jl")
include("models/HawkesProcess.jl")
include("models/DiscreteHawkesProcess.jl")
include("models/Network.jl")
export PoissonProcess,
       HomogeneousProcess,
       LinearProcess,
       ExponentialProcess,
       LogitNormalProcess,
       HawkesProcess,
       rand,
       loglikelihood,
       intensity,
       impulse_response,
       stability,
       HawkesProcess,
       StandardHawkesProcess,
       NetworkHawkesProcess,
       DiscreteHawkesProcess,
       Network,
       DenseNetwork,
       BernoulliNetwork,
       StochasticBlockNetwork

include("inference/mcmc/HawkesProcess.jl")
include("inference/mcmc/DiscreteHawkesProcess.jl")
include("inference/mcmc/Network.jl")
include("inference/vb/DiscreteHawkesProcess.jl")
include("inference/vb/Network.jl")
export mcmc, vb

end # module
