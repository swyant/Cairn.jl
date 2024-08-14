using SafeTestsets

@safetestset "System and Data Tests" begin include("data_tests.jl") end
# @safetestset "QoI Int. Tests" begin include("qoi_integration_tests.jl") end

@safetestset "Basic Committee Potential Tests" begin include("committee_potential_tests.jl") end
@safetestset "Committee QoI Tests" begin include("committee_qoi_tests.jl") end


