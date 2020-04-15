module PointProcesses

using Distributions, Distributed, Gadfly, Printf
import Base.rand

include("utils/normal_gamma.jl")
export NormalGamma

include("utils/assert.jl")
export assert_nonnegative,
       assert_distribution,
       assert_probability

include("models/poisson.jl")
export PoissonProcess,
       HomogeneousProcess,
       LinearProcess,
       ExponentialProcess,
       LogitNormalProcess

include("models/hawkes.jl")
export HawkesProcess,
       NetworkHawkesProcess,
       ExponentialHawkesProcess

end # module
