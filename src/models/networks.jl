abstract type Network end

function rand(m::Network) end


"""
    A network with i.i.d Bernoulli connections: A[n, m] ~ Bernoulli(ρ).
"""
struct BernoulliNetwork
    α
    β
    N
end

function rand(m::BernoulliNetwork)
    ρ = rand(Beta(m.α, m.β))
    return rand(Bernoulli.(ρ), m.N, m.N)
end

function rand(m::BernoulliNetwork, n::Int32)
    data = zeros(m.N, m.N, n)
    for i = 1:n
        data[:, :, i] = rand(m)
    end
    return data
end


"""
    A network with conditionally i.i.d Bernoulli connections.

Each node is associated with a latent "block". Connections from block `i` to block `j` are i.i.d Bernoulli distributed. Specifically,

    (model here...)
"""
struct StochasticBlockNetwork
    α
    β
    γ
    N
end

function rand(m::StochasticBlockNetwork)
    K = length(m.γ)
    π = rand(Dirichlet(m.γ))
    z = dense(rand(Discrete(π), m.N))
    ρ = rand(Beta(m.α, m.β), K, K)
    A = zeros(m.N, m.N)
    for i = 1:m.N
        for j = 1:m.N
            zi = z[i]
            zj = z[j]
            A[i, j] = rand.(Bernoulli.(ρ[zi, zj]))
        end
    end
    return A
end

function rand(m::StochasticBlockNetwork, n::Int32)
    data = zeros(m.N, m.N, n)
    for i = 1:n
        data[:, :, i] = rand(m)
    end
    return data
end

dense(onehots) = vec([idx.I[1] for idx in argmax(onehots, dims=1)])
