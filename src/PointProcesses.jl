module PointProcesses

using Distributions, Distributed, SharedArrays, Gadfly, Printf
import Base.rand

include("utils/distributions.jl")
include("utils/assertions.jl")

include("models/poisson.jl")
include("models/HawkesProcess.jl")
include("models/NetworkHawkesProcess.jl")
include("models/BernoulliNetworkHawkesProcess.jl")
include("models/StochasticBlockNetworkHawkesProcess.jl")
include("models/ExponentialHawkes.jl")
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
       NetworkHawkesProcess,
       BernoulliNetworkHawkesProcess,
       StochasticBlockNetworkHawkesProcess,
       ExponentialHawkesProcess

include("inference/mcmc/mcmc.jl")
include("inference/mcmc/NetworkHawkesProcess.jl")
include("inference/mcmc/BernoulliNetworkHawkesProcess.jl")
include("inference/mcmc/StochasticBlockNetworkHawkesProcess.jl")
include("inference/mcmc/ExponentialHawkesProcess.jl")
export mcmc

end # module
