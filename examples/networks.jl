using Gadfly

n = 1000

network = BernoulliNetwork(1., 1., 10)
data = zeros(m.N, m.N, n)
for i = 1:n
    data[:, :, i] = rand(m)
end
heatmap(mean(data, dims=3))


network = StochasticBlockNetwork(1., 1., ones(3), 10)
data = zeros(m.N, m.N, n)
for i = 1:n
    data[:, :, i] = rand(m)
end
heatmap(mean(data, dims=3))


"""
    A 2-dimensional heatmap.

Based on `Gadfly.spy`.
"""
function heatmap(X)
    n,m = size(X)
    idx = findall(x -> !isnan(x), X)
    values = X[idx]
    plot(x=getdim.(idx, 1), y=getdim.(idx, 2), color=values, Coord.cartesian(yflip=true, fixed=true, xmin=0.5, xmax=m+.5, ymin=0.5, ymax=n+0.5), Geom.rectbin, Scale.x_continuous, Scale.y_continuous)
end

getdim(idx::CartesianIndex, dim) = idx.I[dim]
