module Model

using Distributions
include("MCMC/MCMC.jl")

mutable struct State
  b0::Float64
  b1::Float64
  b2::Float64
end
State() = State(0.0, 0.0, 0.0)

struct Data
  x::Vector{Float64}
  y::Vector{Float64}
  N::Int
  function Data(x, y)
    @assert length(x) == length(y)
    new(x, y, length(x))
  end
end

struct Priors
  b0::Normal
  b1::Normal
  b2::Normal
end
Priors() = Priors(Normal(0, 10), Normal(0, 10), Normal(0, 10))

struct Tuners
  b0::MCMC.TuningParam{Float64}
  b1::MCMC.TuningParam{Float64}
  b2::MCMC.TuningParam{Float64}
end
Tuners() = Tuners(MCMC.TuningParam(.1), MCMC.TuningParam(.1), MCMC.TuningParam(.1))

function update_b0(s::State, d::Data, priors::Priors, tuners::Tuners)
  function log_prob(b0)
    p = MCMC.sigmoid.(b0 .+ s.b1 * d.x + s.b2 * d.x .^ 2)
    loglike = sum(d.y .* log.(p) + (1 .- d.y) .* log1p.(-p))
    logprior = logpdf(priors.b0, b0)
    return loglike + logprior
  end
  s.b0 = MCMC.metropolisAdaptive(s.b0, log_prob, tuners.b0)
end

function update_b1(s::State, d::Data, priors::Priors, tuners::Tuners)
  function log_prob(b1)
    p = MCMC.sigmoid.(s.b0 .+ b1 * d.x + s.b2 * d.x .^ 2)
    loglike = sum(d.y .* log.(p) + (1 .- d.y) .* log1p.(-p))
    logprior = logpdf(priors.b1, b1)
    return loglike + logprior
  end
  s.b1 = MCMC.metropolisAdaptive(s.b1, log_prob, tuners.b1)
end

function update_b2(s::State, d::Data, priors::Priors, tuners::Tuners)
  function log_prob(b2)
    p = MCMC.sigmoid.(s.b0 .+ s.b1 * d.x + b2 * d.x .^ 2)
    loglike = sum(d.y .* log.(p) + (1 .- d.y) .* log1p.(-p))
    logprior = logpdf(priors.b2, b2)
    return loglike + logprior
  end
  s.b2 = MCMC.metropolisAdaptive(s.b2, log_prob, tuners.b1)
end


function fit(data; priors=Priors(), tuners=Tuners(), init=nothing, niter=1000,
             burn=1000, print_every=100)
  if init == nothing 
    state = State()
  end

  out = []
  for iter in 1:(niter + burn)
    if iter % print_every == 0
      print("\r$(iter)/$(niter + burn)")
    end

    update_b0(state, data, priors, tuners)
    update_b1(state, data, priors, tuners)
    update_b2(state, data, priors, tuners)

    if iter > burn
      append!(out, [deepcopy(state)])
    end
  end
  println()

  return out
end

function readsimdat(path="../../dat/dat.txt")
  x = Float64[]
  y = Float64[]
  p = Float64[]
  open(path, "r") do file
    lines = readlines(file)
    for line in lines[2:end]
      xn, yn, pn = parse.(Float64, split(line, ","))
      append!(x, xn)
      append!(y, yn)
      append!(p, pn)
    end
  end

  return Dict(:x => x, :y => y, :p => p)
end

end # Model

### MAIN ###
using Distributions
using PyPlot
const plt = PyPlot.plt

quantiles(x, p; dims) = mapslices(xd -> quantile(xd, p), x, dims=dims)

simdat = Model.readsimdat()
_ = Model.fit(Model.Data(simdat[:x], simdat[:y]), burn=1, niter=1);
@time out = Model.fit(Model.Data(simdat[:x], simdat[:y]), burn=1000);
println("Done")

B = length(out)
b0 = reshape([s.b0 for s in out], B, 1)
b1 = reshape([s.b1 for s in out], B, 1)
b2 = reshape([s.b2 for s in out], B, 1)
M = 200
x = reshape(range(-4, stop=4, length=M), 1, M)
p = Model.MCMC.sigmoid.(b0 .+ b1 .* x .+ b2 .* x .^ 2)
plt.plot(vec(x), vec(mean(p, dims=1)))
plt.plot(vec(x), vec(quantiles(p, .975, dims=1)), linestyle="--")
plt.plot(vec(x), vec(quantiles(p, .025, dims=1)), linestyle="--")
plt.scatter(simdat[:x][1:100:end], simdat[:p][1:100:end], s=5)
plt.scatter(simdat[:x], simdat[:y] + randn(length(simdat[:y]))*.01, s=5, alpha=.1)
plt.show()

