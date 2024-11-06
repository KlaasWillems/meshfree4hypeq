my_tests = ["ParticleGridsTest.jl", "InterpolationsTest.jl"]

println("Running tests:")
for my_test in my_tests
  include(my_test)
end
