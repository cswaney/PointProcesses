# Homogeneous Poisson
p = PointProcess(1.0)
ts = rand(p)

# Inhomogeneous Poisson

# A. Exponential
h = Exponential()
p = InhomogeneousPointProcess(h, 1.)
ts = rand(p, 3)

# B. Logistic-Normal
h = LogitNormal()
p = InhomogeneousPointProcess(h, 1.)
ts = rand(p)
