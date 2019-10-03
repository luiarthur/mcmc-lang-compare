module Model

using Distributions
include("MCMC/MCMC.jl")


log_prob(x::Float64) = logpdf(Normal(0, 1), x)

function f(x, n)
  for _ in 1:n
    x = MCMC.metropolis(x, log_prob, 1.0)
  end
  return x
end

# Compile
f(1.0, 1);

# Time me for real
@time println("x = $(f(1.0, 2e6))")

end
