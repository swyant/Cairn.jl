using Cairn
using Molly
using PotentialLearning, InteratomicPotentials
using Statistics
using SpecialPolynomials: Jacobi
using LinearAlgebra: norm
using Test

pce_template = PolynomialChaos(3, 2, Jacobi{0.5,0.5})
ref = MullerBrownRot()

all_pce_params = [[48.7321 95.9563 11.8879 3.56038 94.8206 51.6035 69.2488 37.2358 81.8482 40.592];
              [81.6018 30.8123 56.9185 98.2884 17.8753 2.37024 65.9343 12.8787 74.2257 12.8148];
              [6.65932 62.8262 0.61548 91.5065 76.1619 27.3688 91.6777 66.1589 34.1041 8.07007];
              [55.9341 15.572  19.6692 95.2555 5.81737 20.1318 68.1779 88.3704 6.86962 30.4034];]
all_pces = [let
                coeffs = all_pce_params[i,:]
                pce = deepcopy(pce_template)
                pce.params = coeffs
                pce
            end
            for i in 1:4]
pce_cmte = CommitteePotential(all_pces,1,u"kJ * mol^-1",u"nm")
mb_sys = System(ref,[0.096079218176701, -0.9623723102484034])


ace_template =  ACE(species            = [:Hf, :O],
                    body_order         = 2,
                    polynomial_degree  = 3,
                    wL                 = 1.5,
                    csp                = 1.0,
                    r0                 = 2.15,
                    rcutoff            = 5.0)

all_ace_params =  [[9.70601 -2.65956 -0.355954 -12.9455 5.23402 -0.736636 -12.9455 5.234   -0.73665  476.549 174.991 39.3035];
                  [8.78225 -3.08078 -0.315163 -12.7109 5.45832 -0.755008 -12.7109 5.45833 -0.755008 468.413 171.969 38.6202];
                  [9.61689 -2.45081 -0.171406 -12.4746 5.66575 -0.629782 -12.4746 5.66575 -0.629779 415.392 153.722 36.1808];
                  [9.43579 -2.65284 -0.257129 -13.0829 5.23399 -0.792417 -13.0829 5.23399 -0.792411 599.668 219.678 46.834]]
all_aces = [let
                coeffs = all_ace_params[i,:]
                LBasisPotential(coeffs,[0.0],ace_template)
            end
            for i in 1:4]

ace_cmte = CommitteePotential(all_aces,1,u"eV", u"Å")
ace_force_units = ace_cmte.energy_units / ace_cmte.length_units
ace_sys = get_system(load_data("./dummy_hfo2.xyz", ExtXYZ(u"eV", u"Å"))[1]) # Should be a FlexibleSystem

function num_excess(array, thresh)
   return count(array .> thresh)
end

function any_excess(array, thresh)
   return any(array .> thresh)
end

function num_abs_excess(array,thresh)
    return count(abs.(array) .> thresh)
end

function norm_then_bool(array,thresh)
    arr_norm = norm(array)
    return (arr_norm > thresh)
end
# bad because reduce fn should reduce to float, integer, or boolean
function bad_function1(array)
    return "strings not allowed"
end

# bad because reduce fn should reduce to single value
function bad_function2(array)
    return 2*array
end

# bad because reduce fn should reduce to single value
function bad_function3(array)
    return [array array]
end

function select_firstentry(array)
    return getindex(array,1)
end

@testset "Committee QoIs" begin
    @testset "CmteEnergy tests" begin
        cmte_energy1 = CmteEnergy()
        energies1 = compute(cmte_energy1,mb_sys,pce_cmte)
        @test typeof(energies1) <: Vector{typeof(1.0 * pce_cmte.energy_units)}
        @test energies1 ≈ [16.894988694723455,
                           174.92114032265883,
                           4.6978599790506435,
                           -88.72911646619953] * pce_cmte.energy_units

        cmte_energy2 = CmteEnergy(;strip_units=true)
        energies2 = compute(cmte_energy2,mb_sys,pce_cmte)
        @test typeof(energies2) <: Vector{<:Real}
        @test energies2 ≈ [16.894988694723455,
                           174.92114032265883,
                           4.6978599790506435,
                           -88.72911646619953]

        cmte_energy3 = CmteEnergy(Statistics.std)
        energy_std = compute(cmte_energy3,mb_sys,pce_cmte)
        @test typeof(energy_std) <: typeof(1.0 * pce_cmte.energy_units)
        @test energy_std ≈ 109.35169147516591 * pce_cmte.energy_units

        cmte_energy4 = CmteEnergy(Statistics.std;strip_units=true)
        energy_std2 = compute(cmte_energy4,mb_sys,pce_cmte)
        @test typeof(energy_std2) <: Real
        @test energy_std2 ≈ 109.35169147516591

        # one aspect of this test is that it checks that the _check_reduction_fn works with unit'ed arrays
        cmte_energy5 = CmteEnergy(enrgs -> num_excess(enrgs,0.0u"kJ*mol^-1"))
        num_excess_energies = compute(cmte_energy5,mb_sys,pce_cmte)
        @test typeof(num_excess_energies) <: Integer
        @test num_excess_energies == 3

        cmte_energy6 = CmteEnergy(enrgs -> any_excess(enrgs,0.0);strip_units=true)
        excess = compute(cmte_energy6,mb_sys,pce_cmte)
        @test typeof(excess) <: Bool
        @test excess == true

        @test_throws "Reduction function invalid. Must reduce to" CmteEnergy(bad_function1)
        @test_throws "Reduction function invalid. Must reduce to" CmteEnergy(bad_function2)
        @test_throws "Reduction function invalid. Must reduce to" CmteEnergy(bad_function3)
    end

    #### CmteFlatForces tests
    #### for the forces, switching to ACE because want to have more than one atom

    @testset "CmteFlatForces tests" begin
        cmte_fforces1 = CmteFlatForces()
        flatforces1 = compute(cmte_fforces1,ace_sys,ace_cmte)
        @test typeof(flatforces1) <: Matrix{typeof(1.0*ace_force_units)}
        @test flatforces1 ≈ [[-6.37709608714023 0.036413014433861475 0.24146709578677694 6.37709608714023 -0.036413014433861475 -0.24146709578677694];
                              [-6.38380632343097 0.034489607530542365 0.20447915531187807 6.38380632343097 -0.034489607530542365 -0.20447915531187807];
                              [-6.52462885442848 0.038876665247638550 0.25392612216268523 6.52462885442848 -0.038876665247638550 -0.25392612216268523];
                              [-6.37029042242475 0.034663551068511725 0.21774432384714262 6.37029042242475 -0.034663551068511725 -0.21774432384714262]] * ace_force_units

        cmte_fforces2 = CmteFlatForces((cmte=Statistics.std,))
        # check that this NamedTuple-based constructor set things up properly
        @test cmte_fforces2.cmte_reduce == Statistics.std
        @test isnothing(cmte_fforces2.coord_and_atom_reduce)
        @test cmte_fforces2.reduce_order == [1]

        flatforces2 = compute(cmte_fforces2,ace_sys,ace_cmte)
        @test typeof(flatforces2) <: Vector{typeof(1.0*ace_force_units)}
        @test size(flatforces2) == (6,)
        @test flatforces2 == [0.07398833138686434,
                               0.0020383112775153476,
                               0.022390836562588386,
                               0.07398833138686434,
                               0.0020383112775153476,
                               0.022390836562588386] * ace_force_units

        # sanity check that strip_units works
        cmte_fforces2_strip = CmteFlatForces((cmte=Statistics.std,); strip_units=true)
        flatforces2_strip = compute(cmte_fforces2_strip,ace_sys,ace_cmte)
        @test typeof(flatforces2_strip) <: Vector{Float64}
        @test flatforces2_strip == [0.07398833138686434,
                               0.0020383112775153476,
                               0.022390836562588386,
                               0.07398833138686434,
                               0.0020383112775153476,
                               0.022390836562588386]

        cmte_fforces3 = CmteFlatForces((coord_and_atom=maximum,))
        @test isnothing(cmte_fforces3.cmte_reduce)
        @test cmte_fforces3.coord_and_atom_reduce == maximum
        @test cmte_fforces3.reduce_order == [2]

        flatforces3 = compute(cmte_fforces3,ace_sys,ace_cmte)
        @test typeof(flatforces3) <: Vector{typeof(1.0*ace_force_units)}
        @test size(flatforces3) == (4,)
        @test flatforces3  == [6.37709608714023,
                               6.383806323430971,
                               6.524628854428479,
                               6.37029042242475] * ace_force_units


        cmte_fforces3_strip = CmteFlatForces((coord_and_atom=maximum,); strip_units=true)
        flatforces3_strip = compute(cmte_fforces3_strip,ace_sys,ace_cmte)
        @test typeof(flatforces3_strip) <: Vector{Float64}
        @test flatforces3_strip  == [6.37709608714023,
                                     6.383806323430971,
                                     6.524628854428479,
                                     6.37029042242475]


        cmte_fforces4 = CmteFlatForces((cmte=Statistics.std, coord_and_atom=Statistics.mean))
        @test cmte_fforces4.cmte_reduce == Statistics.std
        @test cmte_fforces4.coord_and_atom_reduce == Statistics.mean
        @test cmte_fforces4.reduce_order == [1,2]

        flatforces4 = compute(cmte_fforces4,ace_sys,ace_cmte)
        @test typeof(flatforces4) <: typeof(1.0*ace_force_units)
        @test flatforces4 == 0.03280582640898936 * ace_force_units

        cmte_fforces5 = CmteFlatForces((coord_and_atom=maximum, cmte=Statistics.mean))
        @test cmte_fforces5.cmte_reduce == Statistics.mean
        @test cmte_fforces5.coord_and_atom_reduce == maximum
        @test cmte_fforces5.reduce_order == [2,1]

        flatforces5 = compute(cmte_fforces5,ace_sys,ace_cmte)
        @test typeof(flatforces5) <: typeof(1.0*ace_force_units)
        @test flatforces5 == 6.413955421856107 * ace_force_units

        ##### Integer/Boolean outputs, focusing on computes

        cmte_fforces6 = CmteFlatForces((coord_and_atom=(x->num_abs_excess(x,6.5)),); strip_units=true)
        flatforces6 = compute(cmte_fforces6,ace_sys,ace_cmte)
        @test typeof(flatforces6) <: Vector{Int64}
        @test flatforces6 == [0,0,2,0]

        #i.e., does any cmte predict forces above 6.5 eV/Å
        cmte_fforces7 = CmteFlatForces((coord_and_atom=(x->num_abs_excess(x,6.5)),
                                        cmte=(y->any_excess(y,0)));
                                        strip_units=true)
        flatforces7 = compute(cmte_fforces7,ace_sys,ace_cmte)
        @test typeof(flatforces7) <: Bool
        @test flatforces7 == true

        cmte_fforces8 = CmteFlatForces((coord_and_atom=(x->any_excess(x,6.0)),); strip_units=true)
        flatforces8 = compute(cmte_fforces8,ace_sys,ace_cmte)
        @test typeof(flatforces8) <: Vector{Bool}
        @test flatforces8 == [true,true,true,true]

        #i.e., how many cmtes predict forces above 6.0 eV/Å
        cmte_fforces9 = CmteFlatForces((coord_and_atom=(x->any_excess(x,6.0)),
                                        cmte=count);
                                        strip_units=true)
        flatforces9 = compute(cmte_fforces9,ace_sys,ace_cmte)
        @test typeof(flatforces9) <: Int64
        @test flatforces9 == 4

        #i.e. how many force components have avg values above 6.0 eV/Å
        cmte_fforces10 = CmteFlatForces((cmte=Statistics.mean,
                                         coord_and_atom=(x->num_abs_excess(x,6.0)));
                                         strip_units=true)
        flatforces10 = compute(cmte_fforces10,ace_sys,ace_cmte)
        @test typeof(flatforces10) <: Int64
        @test flatforces10 == 2

        @test_throws """Only allowed keys are 'cmte', 'coord_and_atom' """ CmteFlatForces((wrong_key=maximum,))
        @test_throws """Only allowed keys are 'cmte', 'coord_and_atom' """ CmteFlatForces((wrong_key=maximum, cmte=maximum))
        @test_throws """Only allowed keys are 'cmte', 'coord_and_atom' """ CmteFlatForces((coord_and_atom=maximum, cmte=Statistics.mean, wrong_key=maximum))
        #@test_throws "There can be a maximum of 2 elements in the passed NamedTuple" CmteFlatForces((coord_and_atom=maximum, cmte=Statistics.mean, cmte=maximum))

        @test_throws "Cmte reduction function invalid. Must reduce"  CmteFlatForces((cmte=bad_function1,))
        @test_throws "Cmte reduction function invalid. Must reduce" CmteFlatForces((cmte=bad_function2,))
        @test_throws "Cmte reduction function invalid. Must reduce" CmteFlatForces((cmte=bad_function1, coord_and_atom=maximum))
        @test_throws "Cmte reduction function invalid. Must reduce" CmteFlatForces((coord_and_atom=maximum, cmte=bad_function1))
        @test_throws "Coord_and_atom reduction function invalid. Must reduce" CmteFlatForces((coord_and_atom=bad_function1,))
        @test_throws "Coord_and_atom reduction function invalid. Must reduce" CmteFlatForces((coord_and_atom=bad_function1,cmte=maximum))
        @test_throws "Coord_and_atom reduction function invalid. Must reduce" CmteFlatForces((cmte=maximum, coord_and_atom=bad_function1))
        @test_throws "Cmte reduction and coord_and_atom reduction functions invalid. Must reduce" CmteFlatForces((cmte=bad_function1, coord_and_atom=bad_function1))
        @test_throws "Cmte reduction and coord_and_atom reduction functions invalid. Must reduce" CmteFlatForces((cmte=bad_function1, coord_and_atom=bad_function2))
        @test_throws "Cmte reduction and coord_and_atom reduction functions invalid. Must reduce" CmteFlatForces((coord_and_atom=bad_function1, cmte=bad_function2))

        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteFlatForces(maximum,maximum,[0,1],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteFlatForces(maximum,maximum,[1,3],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteFlatForces(maximum,maximum,[1,2,2],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteFlatForces(maximum,nothing,[1,2],false)
    end

    @testset "CmteForces tests" begin
        cmte_forces1 = CmteForces()
        forces1 = compute(cmte_forces1,ace_sys,ace_cmte)
        #TODO this is specific to current IP.jl, and is subject to change when it is fully AC-compatible
        @test typeof(forces1) <: Vector{Vector{Vector{typeof(1.0*ace_force_units)}}}
        ref_forces1 = [[[-6.37709608714023,0.036413014433861475,0.24146709578677694 ],[6.37709608714023,-0.036413014433861475,-0.24146709578677694]],
                       [[-6.383806323430971,0.034489607530542365,0.20447915531187807],[6.383806323430971,-0.034489607530542365,-0.20447915531187807]],
                       [[-6.524628854428479,0.03887666524763855,0.25392612216268523 ],[6.524628854428479,-0.03887666524763855,-0.25392612216268523]],
                       [[-6.37029042242475,0.034663551068511725,0.21774432384714262 ],[6.37029042242475,-0.034663551068511725,-0.21774432384714262]]] * ace_force_units
        @test all( [all(forces1[i] .≈ ref_forces1[i]) for i in eachindex(forces1)] )

        cmte_forces1_strip = CmteForces(;strip_units=true)
        forces1_strip = compute(cmte_forces1_strip,ace_sys,ace_cmte)
        @test typeof(forces1_strip) <: Vector{Vector{Vector{Float64}}}

        cmte_forces2 = CmteForces((cmte=Statistics.std,))
        @test cmte_forces2.cmte_reduce == Statistics.std
        @test isnothing(cmte_forces2.atom_reduce)
        @test isnothing(cmte_forces2.coord_reduce)
        @test cmte_forces2.reduce_order == [1]

        forces2 = compute(cmte_forces2,ace_sys,ace_cmte)
        @test typeof(forces2) <: Matrix{typeof(1.0*ace_force_units)}
        @test size(forces2) == (2,3)
        @test forces2 ≈ [[0.07398833138686434 0.0020383112775153476 0.022390836562588386];
                         [0.07398833138686434 0.0020383112775153476 0.022390836562588386]] * ace_force_units

        cmte_forces2_strip = CmteForces((cmte=Statistics.std,); strip_units=true)
        forces2_strip = compute(cmte_forces2_strip,ace_sys,ace_cmte)
        @test typeof(forces2_strip) <: Matrix{Float64}
        @test forces2_strip ≈ [[0.07398833138686434 0.0020383112775153476 0.022390836562588386];
                               [0.07398833138686434 0.0020383112775153476 0.022390836562588386]]

        # will be zero for all conservative potentials
        cmte_forces3 = CmteForces((atom=sum,))
        @test isnothing(cmte_forces3.cmte_reduce)
        @test cmte_forces3.atom_reduce == sum
        @test isnothing(cmte_forces3.coord_reduce)
        @test cmte_forces3.reduce_order == [2]

        forces3 = compute(cmte_forces3, ace_sys, ace_cmte)
        @test typeof(forces3) <: Matrix{typeof(1.0*ace_force_units)}
        @test size(forces3) == (4,3)
        ref_forces3 = zeros(Float64,4,3) *ace_force_units
        @test isapprox(forces3, ref_forces3; atol=eps(Float64)*ace_force_units)

        cmte_forces3_strip = CmteForces((atom=sum,);strip_units=true)
        forces3_strip = compute(cmte_forces3_strip,ace_sys,ace_cmte)
        @test typeof(forces3_strip) <: Matrix{Float64}
        ref_forces3_strip = zeros(Float64,4,3)
        @test isapprox(forces3_strip, ref_forces3_strip; atol=eps(Float64))

        cmte_forces4 =  CmteForces((coord=norm,))
        @test isnothing(cmte_forces4.cmte_reduce)
        @test isnothing(cmte_forces4.atom_reduce)
        @test cmte_forces4.coord_reduce == norm
        @test cmte_forces4.reduce_order == [3]

        forces4 = compute(cmte_forces4,ace_sys,ace_cmte)
        @test typeof(forces4) <: Matrix{typeof(1.0*ace_force_units)}
        @test size(forces4) == (4,2)
        @test forces4 == [[6.381769877595642 6.381769877595642]; [6.3871734306390895 6.3871734306390895];
                          [6.52968387892232 6.52968387892232]; [6.3741049739062845 6.3741049739062845]] * ace_force_units

        cmte_forces4_strip =  CmteForces((coord=norm,);strip_units=true)
        forces4_strip = compute(cmte_forces4_strip,ace_sys,ace_cmte)
        @test typeof(forces4_strip) <: Matrix{Float64}
        @test forces4_strip == [[6.381769877595642 6.381769877595642]; [6.3871734306390895 6.3871734306390895];
                                [6.52968387892232 6.52968387892232]; [6.3741049739062845 6.3741049739062845]]

        cmte_forces5 = CmteForces((cmte=Statistics.std,atom=maximum))
        @test cmte_forces5.cmte_reduce == Statistics.std
        @test cmte_forces5.atom_reduce == maximum
        @test isnothing(cmte_forces5.coord_reduce)
        @test cmte_forces5.reduce_order == [1,2]

        forces5 = compute(cmte_forces5, ace_sys, ace_cmte)
        @test typeof(forces5) <: Vector{typeof(1.0*ace_force_units)}
        @test size(forces5) == (3,)
        @test forces5 ≈ [0.07398833138686434, 0.0020383112775153476, 0.022390836562588386] * ace_force_units

        cmte_forces6 = CmteForces((cmte=Statistics.mean, coord=norm))
        @test cmte_forces6.cmte_reduce == Statistics.mean
        @test isnothing(cmte_forces6.atom_reduce)
        @test cmte_forces6.coord_reduce == norm
        @test cmte_forces6.reduce_order == [1,3]

        forces6 = compute(cmte_forces6, ace_sys, ace_cmte)
        @test typeof(forces6) <: Vector{typeof(1.0*ace_force_units)}
        @test size(forces6) == (2,)
        @test forces6 ≈ [6.41815817911017,6.41815817911017] * ace_force_units

        cmte_forces7 = CmteForces((atom=maximum, cmte=Statistics.mean))
        @test cmte_forces7.cmte_reduce == Statistics.mean
        @test cmte_forces7.atom_reduce == maximum
        @test isnothing(cmte_forces7.coord_reduce)
        @test cmte_forces7.reduce_order == [2,1]

        forces7 = compute(cmte_forces7, ace_sys, ace_cmte)
        @test typeof(forces7) <: Vector{typeof(1.0*ace_force_units)}
        @test size(forces7) == (3,)
        @test forces7 ≈[6.413955421856107,0.036110709570138524,0.22940417427712073]  * ace_force_units

        cmte_forces8 = CmteForces((atom=maximum,
                                   coord=select_firstentry))
        @test isnothing(cmte_forces8.cmte_reduce)
        @test cmte_forces8.atom_reduce == maximum
        @test cmte_forces8.coord_reduce == select_firstentry
        @test cmte_forces8.reduce_order == [2,3]

        forces8 = compute(cmte_forces8, ace_sys, ace_cmte)
        @test typeof(forces8) <: Vector{typeof(1.0*ace_force_units)}
        @test size(forces8) == (4,)
        @test forces8 ≈ [6.37709608714023,
                         6.383806323430971,
                         6.524628854428479,
                         6.37029042242475]  * ace_force_units

        cmte_forces9 = CmteForces((coord=norm,
                                   cmte=Statistics.std))
        @test cmte_forces9.cmte_reduce == Statistics.std
        @test isnothing(cmte_forces9.atom_reduce)
        @test cmte_forces9.coord_reduce == norm
        @test cmte_forces9.reduce_order == [3,1]

        forces9 = compute(cmte_forces9, ace_sys, ace_cmte)
        @test typeof(forces9) <: Vector{typeof(1.0*ace_force_units)}
        @test size(forces9) == (2,)
        @test forces9 ≈ [0.07452701358108477,0.07452701358108477]* ace_force_units

        cmte_forces10 = CmteForces((coord=norm,
                                    atom=Statistics.var))
        @test isnothing(cmte_forces10.cmte_reduce)
        @test cmte_forces10.atom_reduce == Statistics.var
        @test cmte_forces10.coord_reduce == norm
        @test cmte_forces10.reduce_order == [3,2]

        forces10 = compute(cmte_forces10, ace_sys, ace_cmte)
        @test typeof(forces10) <: Vector{typeof(1.0*ace_force_units^2)}
        @test size(forces10) == (4,)
        @test isapprox(forces10,zeros(Float64,4)* ace_force_units^2;
                       atol=eps(Float64)*ace_force_units^2)


        cmte_forces11 = CmteForces((cmte=Statistics.std,atom=maximum,coord=select_firstentry))
        @test cmte_forces11.cmte_reduce == Statistics.std
        @test cmte_forces11.atom_reduce == maximum
        @test cmte_forces11.coord_reduce == select_firstentry
        @test cmte_forces11.reduce_order == [1,2,3]

        forces11 = compute(cmte_forces11, ace_sys, ace_cmte)
        @test typeof(forces11) <: typeof(1.0*ace_force_units)
        @test forces11 ≈  0.07398833138686434* ace_force_units

        cmte_forces12 = CmteForces((cmte=Statistics.mean,coord=norm,atom=maximum,))
        @test cmte_forces12.cmte_reduce == Statistics.mean
        @test cmte_forces12.atom_reduce == maximum
        @test cmte_forces12.coord_reduce == norm
        @test cmte_forces12.reduce_order == [1,3,2]

        forces12 = compute(cmte_forces12, ace_sys, ace_cmte)
        @test typeof(forces12) <: typeof(1.0*ace_force_units)
        @test forces12 ≈ 6.41815817911017* ace_force_units

        cmte_forces13 = CmteForces((atom=maximum,coord=norm,cmte=Statistics.std))
        @test cmte_forces13.cmte_reduce == Statistics.std
        @test cmte_forces13.atom_reduce == maximum
        @test cmte_forces13.coord_reduce == norm
        @test cmte_forces13.reduce_order == [2,3,1]

        forces13 = compute(cmte_forces13, ace_sys, ace_cmte)
        @test typeof(forces13) <: typeof(1.0*ace_force_units)
        @test forces13 ≈ 0.07452701358108477 * ace_force_units

        cmte_forces14 = CmteForces((coord=norm,cmte=Statistics.std,atom=sum))
        @test cmte_forces14.cmte_reduce == Statistics.std
        @test cmte_forces14.atom_reduce == sum
        @test cmte_forces14.coord_reduce == norm
        @test cmte_forces14.reduce_order == [3,1,2]

        forces14 = compute(cmte_forces14, ace_sys, ace_cmte)
        @test typeof(forces14) <: typeof(1.0*ace_force_units)
        @test forces14 ≈ 0.14905402716216953 * ace_force_units

        cmte_forces15 = CmteForces((coord=norm,atom=sum,cmte=Statistics.std))
        @test cmte_forces15.cmte_reduce == Statistics.std
        @test cmte_forces15.atom_reduce == sum
        @test cmte_forces15.coord_reduce == norm
        @test cmte_forces15.reduce_order == [3,2,1]

        forces15 = compute(cmte_forces15, ace_sys, ace_cmte)
        @test typeof(forces15) <: typeof(1.0*ace_force_units)
        @test forces15 ≈ 0.14905402716216953 * ace_force_units #this would be different than forces14 if there were more than two atoms

        ## Integer and Boolean tests
        cmte_forces16 = CmteForces((cmte=(x->num_abs_excess(x,6.38)),);strip_units=true)
        forces16 = compute(cmte_forces16, ace_sys, ace_cmte)
        @test typeof(forces16) <: Matrix{<:Integer}
        @test size(forces16) == (2,3)
        @test forces16 == [[2 0 0]; [2 0 0 ]]

        cmte_forces17 = CmteForces((cmte=(x->any_excess(x,6.38)),);strip_units=true)
        forces17 = compute(cmte_forces17, ace_sys, ace_cmte)
        @test typeof(forces17) <: Matrix{Bool}
        @test size(forces17) == (2,3)
        @test forces17 == [[false false false]; [true false false]]

        cmte_forces18 = CmteForces((atom=(x->num_abs_excess(x,6.38)),);strip_units=true)
        forces18 = compute(cmte_forces18, ace_sys, ace_cmte)
        @test typeof(forces18) <: Matrix{<:Integer}
        @test size(forces18) == (4,3)
        @test forces18 == [[0 0 0];[2 0 0];[2 0 0];[0 0 0]]

        cmte_forces19 = CmteForces((coord=(x->norm_then_bool(x,6.38)),);strip_units=true)
        forces19 = compute(cmte_forces19,ace_sys,ace_cmte)
        @test typeof(forces19) <: Matrix{Bool}
        @test size(forces19) == (4,2)
        @test forces19 == [[true true]; [true true]; [true true]; [false false]]

        cmte_forces20 = CmteForces((coord=norm,atom=(x->num_excess(x,6.38)));strip_units=true)
        forces20 = compute(cmte_forces20,ace_sys,ace_cmte)
        @test typeof(forces20) <: Vector{Int64}
        @test size(forces20) == (4,)
        @test forces20 == [2,2,2,0]

        cmte_forces21 = CmteForces((coord=norm,atom=(x->any_excess(x,6.38)));strip_units=true)
        forces21 = compute(cmte_forces21,ace_sys,ace_cmte)
        @test typeof(forces21) <: Vector{Bool}
        @test size(forces21) == (4,)
        @test forces21 == [true,true,true,false]

        cmte_forces21 = CmteForces((coord=norm,atom=maximum,cmte=(x->num_excess(x,6.38)));strip_units=true)
        forces21 = compute(cmte_forces21,ace_sys,ace_cmte)
        @test typeof(forces21) <: Int64
        @test forces21 == 3

        cmte_forces22 = CmteForces((coord=norm,atom=maximum,cmte=(x->any_excess(x,6.38)));strip_units=true)
        forces22 = compute(cmte_forces22,ace_sys,ace_cmte)
        @test typeof(forces22) <: Bool
        @test forces22 == true

        cmte_forces23 = CmteForces((coord=norm,atom=(x->num_excess(x,6.38)),cmte=mean); strip_units=true)
        forces23 = compute(cmte_forces23,ace_sys,ace_cmte)
        @test typeof(forces23) <: Float64
        @test forces23 ≈ 1.5

        @test_throws """Only allowed keys are 'cmte', 'atom', 'coord' """ CmteForces((wrong_key=maximum,))
        @test_throws """Only allowed keys are 'cmte', 'atom', 'coord' """ CmteForces((wrong_key=maximum, coord=norm))
        @test_throws """Only allowed keys are 'cmte', 'atom', 'coord' """ CmteForces((coord=norm,wrong_key=maximum))
        @test_throws """Only allowed keys are 'cmte', 'atom', 'coord' """ CmteForces((coord=norm,atom=maximum, wrong_key=maximum))
        @test_throws """Only allowed keys are 'cmte', 'atom', 'coord' """ CmteForces((coord=norm, atom=maximum, cmte=Statistics.mean, wrong_key=maximum))

        @test_throws "The cmte reduce function is invalid. Must reduce to"  CmteForces((cmte=bad_function1,))
        @test_throws "The cmte reduce function is invalid. Must reduce to" CmteForces((cmte=bad_function2,))
        @test_throws "The atom reduce function is invalid. Must reduce to"  CmteForces((atom=bad_function1,))
        @test_throws "The coord reduce function is invalid. Must reduce to"  CmteForces((coord=bad_function1,))
        @test_throws "The cmte reduce function is invalid. Must reduce to" CmteForces((cmte=bad_function1, atom=maximum))
        @test_throws "The cmte reduce function is invalid. Must reduce to" CmteForces((coord=norm, cmte=bad_function1))
        @test_throws "The atom reduce function is invalid. Must reduce to" CmteForces((coord=norm, atom=bad_function1))

        @test_throws "The cmte reduce, atom reduce functions are invalid." CmteForces((cmte=bad_function1,atom=bad_function1))
        @test_throws "The cmte reduce, atom reduce functions are invalid." CmteForces((atom=bad_function1,cmte=bad_function1))
        @test_throws "The cmte reduce, coord reduce functions are invalid." CmteForces((coord=bad_function1,cmte=bad_function2))
        @test_throws "The cmte reduce, atom reduce, coord reduce functions are invalid." CmteForces((cmte=bad_function1,
                                                                                                     atom=bad_function1,
                                                                                                     coord=bad_function2))
        @test_throws "The cmte reduce, atom reduce, coord reduce functions are invalid." CmteForces((coord=bad_function1,
                                                                                                     atom=bad_function1,
                                                                                                     cmte=bad_function2))

        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(maximum,maximum,nothing,[0,1],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(maximum,maximum,nothing,[1,4],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(maximum,maximum,nothing,[1,2,2],false)ej
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(maximum,maximum,maximum,[1,2,4],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(maximum,nothing,nothing,[1,2],false)
        @test_throws "Invalid reduce_order. Please use NamedTuple-based constructor" CmteForces(nothing,nothing,nothing,[1],false)
    end

end
