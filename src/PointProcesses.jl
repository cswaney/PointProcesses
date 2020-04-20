module PointProcesses

using Distributions, Distributed, Gadfly, Printf
import Base.rand

include("utils/distributions.jl")
export NormalGamma

include("utils/assertions.jl")
export assert_nonnegative,
       assert_distribution,
       assert_probability

include("models/poisson.jl")
export PoissonProcess,
       HomogeneousProcess,
       LinearProcess,
       ExponentialProcess,
       LogitNormalProcess

include("models/HawkesProcess.jl")
export HawkesProcess,
       rand,
       loglikelihood,
       intensity,
       impulse_response,
       stability

include("models/NetworkHawkesProcess.jl")
export NetworkHawkesProcess

include("inference/mcmc/mcmc.jl")
export mcmc

include("inference/mcmc/NetworkHawkesProcess.jl")

end # module
