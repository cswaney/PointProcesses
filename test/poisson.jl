using Distributions, Gadfly

# Homogeneous
p = PointProcess(1.)
ts = rand(p)

# Inhomogeneous (Exponential)
h = Exponential()
p = PointProcess(1., h)
ts = rand(p)

# Inhomogeneous (Logistic-Normal)
h = LogitNormal()
p = PointProcess(1., h)
ts = rand(p)
