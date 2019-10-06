
module TestJuliaSpeed
using Distributions

mutable struct State
  a
  b::Float64
end


function f1!(s, n::Int)
  x::Float64 = s.b
  for i in 1:n
    for j in 1:n
      x += randn()
    end
  end

  s.b = x
end


# MAIN
s1 = State(randn(3), 1.0)
f1!(s1, 1)
sum(randn(100*100))

@time f1!(s1, 100)
@time sum(randn(100*100))  # slightly slower (high memory consumption)

end  # of module
