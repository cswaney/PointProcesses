@testset "Sufficient Statistics" begin
    # Construct test data
    N = 2
    λ0 = ones(N)
    W = 0.5 * ones(N, N)
    A = ones(N, N)
    μ = zeros(N, N)
    τ = ones(N, N)
    α0 = 1.
    β0 = 1.
    κ = 1.
    ν = ones(N, N)
    μμ = 0.
    κμ = 1.
    ατ = 1.
    βτ = 1.
    Δtmax = 1.
    net = DenseNetwork(N)
    p = NetworkHawkesProcess(λ0, μ, τ, A, W, Δtmax, N, α0, β0, κ, ν, μμ, κμ, ατ, βτ, net)
    # events, nodes = rand(p, 1.)
    events = [0.17763696543418717, 0.3974556075716502, 0.8231761811548723, 1.154166685889635]
    nodes = Int32[2, 2, 2, 1]
    # parent_nodes = PointProcesses.sample_parents(p, events, nodes)
    parent_nodes = [0, 0, 1, 2]

    # Mn
    @test PointProcesses.node_counts(nodes, N) == [1., 3.]

    # Mnm
    @test PointProcesses.parent_counts(nodes, parent_nodes, N) == [0. 1.; 1. 0.]

    # M0
    @test PointProcesses.baseline_counts(nodes, parent_nodes, N) == [0., 2.]

    # xbar
    @test PointProcesses.log_duration(0., 0.5, 1.) == 0.
    # @test log_duration_sum
    # @test log_duration_sum

end
