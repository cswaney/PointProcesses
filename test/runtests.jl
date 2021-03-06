using Test
using Revise
using Pkg
Pkg.activate(".")
using PointProcesses
using Dates

function (≜)(a::T, b::T) where {T}
    fields = fieldnames(T)
    bools = [getfield(a, f) == getfield(b, f) for f in fields]
    return all(bools)
end

# include("poisson.jl")
# include("hawkes.jl")
# include("network_hawkes.jl")
include("mcmc.jl")
