using Cairn
using PotentialLearning: learn!
using Molly #also reexports Unitful
using AtomsCalculators
import SpecialPolynomials: Jacobi
import StaticArrays: SVector
using Test

pce_template = PolynomialChaos(3, 2, Jacobi{0.5,0.5})
ref = MullerBrownRot()

@testset "Basic Committee Potential Tests" begin
    # randomly generated parameters
    all_params = [[48.7321 95.9563 11.8879 3.56038 94.8206 51.6035 69.2488 37.2358 81.8482 40.592];
                  [81.6018 30.8123 56.9185 98.2884 17.8753 2.37024 65.9343 12.8787 74.2257 12.8148];
                  [6.65932 62.8262 0.61548 91.5065 76.1619 27.3688 91.6777 66.1589 34.1041 8.07007];
                  [55.9341 15.572  19.6692 95.2555 5.81737 20.1318 68.1779 88.3704 6.86962 30.4034];
                  [88.3364 62.2294 26.9018 39.6262 46.7406 99.4504 16.5021 38.1385 6.07858 66.3549]]

    all_pces = [let
                    coeffs = all_params[i,:]
                    pce = deepcopy(pce_template)
                    pce.params = coeffs
                    pce
                end
                for i in 1:5]

    ##Checking different constructors
    cmte_pot1 = CommitteePotential(all_pces,3,u"kJ * mol^-1",u"nm")
    @test typeof(cmte_pot1) <: CommitteePotential{<:PolynomialChaos}
    cmte_pot2 = CommitteePotential(all_pces,3,u"eV",u"Å") # units don't match member PCE, but this is not enforced
    @test typeof(cmte_pot2) <: CommitteePotential{<:PolynomialChaos}
    cmte_pot3 = CommitteePotential(all_pces,3,NoUnits,NoUnits) # units don't match member PCE, but this is not enforced
    @test typeof(cmte_pot3) <: CommitteePotential{<:PolynomialChaos}
    cmte_pot4 = CommitteePotential(all_pces;energy_units=u"kJ * mol^-1",length_units= u"nm")
    @test typeof(cmte_pot4) <: CommitteePotential{<:PolynomialChaos}
    @test cmte_pot4.leader == 1
    cmte_pot5 = CommitteePotential(all_pces) #adopts default units, which don't match member PCE units, but this is not enforced
    @test typeof(cmte_pot5) <: CommitteePotential{<:PolynomialChaos}
    @test cmte_pot5.leader == 1
    @test cmte_pot5.energy_units == u"eV"
    @test cmte_pot5.length_units == u"Å"

    #TODO need to check force units as well
    #TODO: redo above and below with a different type of member potential, prefereably ACE

    #Checking bad inputs
    @test_throws "leader index is out of bounds" CommitteePotential(all_pces,7)
    @test_throws "leader index is out of bounds" CommitteePotential(all_pces,-1)
    @test_throws MethodError CommitteePotential(all_pces;energy_units=u"nm") #wrong unit dimensions
    @test_throws MethodError CommitteePotential(all_pces;length_units=u"K") #wrong unit dimensions

    # construct simple reference Molly system
    xref = SVector{2}([0.096079218176701, -0.9623723102484034]) .* u"nm"
    sys_ref = System(ref,xref) # MullerBrown not actually invoked, just a "neutral" potential for these AtomsCalculators tests below

    #TODO use AtomsCalculators test suite

    # checking against reference values that cmte_pot computes correct "leader" forces and energies
    echeck1 = AtomsCalculators.potential_energy(sys_ref,cmte_pot1) #leader is 3
    @test typeof(echeck1) <: typeof(1.0u"kJ * mol^-1")
    @test echeck1 ≈ 4.6978599790506435u"kJ * mol^-1"

    fcheck1 = AtomsCalculators.forces(sys_ref,cmte_pot1) # leader is 3
    @test length(fcheck1) == 1 # only one atom, should test other conditions
    @test typeof(fcheck1) <: Vector{SVector{2,typeof(1.0u"kJ * mol^-1 * nm^-1")}}
    @test fcheck1[1] ≈ [-57.734110333280256, -12.508601878931005]u"kJ * mol^-1 * nm^-1"

    # Molly equivalence test, i.e. running with cmte_pot w/ leader 3 should produce same traj as directly running with pce3
    pce3 = deepcopy(pce_template)
    pce3.params = all_params[3,:]
    init_vels = random_velocities(sys_ref,300.0u"K")
    sys_pce3 = Molly.System(sys_ref;
                            general_inters =(pce3,),
                            velocities=init_vels,
                            force_units=u"kJ * mol^-1 * nm^-1", #want to be explicit with units just in case
                            energy_units=u"kJ * mol^-1")

    sim_nh = NoseHoover(dt=0.002u"ps",          #deterministic simulator
                        temperature=300.0u"K",
                        remove_CM_motion=1)

    simulate!(sys_pce3,sim_nh,100)

    sys_cmtepot1 = Molly.System(sys_ref;
                                general_inters =(cmte_pot1,),
                                velocities=init_vels,
                                force_units=u"kJ * mol^-1 * nm^-1", #want to be explicit with units just in case
                                energy_units=u"kJ * mol^-1")

    simulate!(sys_cmtepot1,sim_nh,100)

    @test position(sys_pce3)[1] ≈ position(sys_cmtepot1)[1]
    @test forces(sys_pce3)[1] ≈ forces(sys_cmtepot1)[1]
end

@testset "Committee learning tests" begin
    #20 train configs
    xtrain = [[0.4519952180711433, 0.5497448756263469],
              [0.43234733814127085, 0.30271181087883803],
              [0.512719924288009, 0.508119011422272],
              [0.3838932703686507, 0.46787859991401054],
              [0.4112084122508261, 0.5834489482382508],
              [0.3733236009121203, 0.5145536221893724],
              [0.3675921743067639, 0.4648583473605157],
              [0.29488055159371873, 0.45832617650052204],
              [0.49583811735652816, 0.5207984179333824],
              [0.48277910519481615, 0.49657706279870967],
              [0.5619272694119924, 0.570496756729312],
              [0.4867075789772271, 0.5069427093912473],
              [0.3717668095607427, 0.42131015100803504],
              [0.45186421974587443, 0.5018415668498282],
              [0.3739717782914152, 0.44139425611430433],
              [0.4322369456805266, 0.4727771278571285],
              [0.3762086855736097, 0.33496173466214246],
              [0.35446473691300173, 0.3077812188010341],
              [0.3808987788084974, 0.45638501845445484],
              [0.5336818950792285, 0.45369548643495206]]

    train_systems = Ensemble(ref,xtrain)
    ilp1 = InefficientLearningProblem(;ref=ref)

    fitted_pce = learn(ilp1,pce_template,train_systems)
    @test length(fitted_pce.params) == 10
    #not sure how robust this test will be, but it would be nice to check that the learn!() is giving meaningful params
    @test fitted_pce.params ≈ [-335.3750515214229,
                                615.4631933197177,
                                247.61997737909118,
                                -351.787767444939,
                                -976.3218308181512,
                                342.07362979908646,
                                -804.7445886595718,
                                2167.8887229673587,
                                -1363.034451052221,
                                539.6457680262732]

    cmte_pot_template = CommitteePotential([deepcopy(pce_template) for _ in 1:10], 7)

    #initial state of commitee LearningProblem
    clp_template = SubsampleAppendCmteRetrain(InefficientLearningProblem(;ref=ref),
                                              [[-1] for _ in 1:10])

    clp1 = deepcopy(clp_template)
    fit_cmte_pot1 = learn!(clp1, cmte_pot_template,train_systems;frac=0.7)
    @test all(length.(clp1.cmte_indices) .== 14) # 0.7*20=14
    @test length(Set([indices for indices in clp1.cmte_indices])) == 10 # ensure that each index sets is unique
    #there are some rare edge cases where this doesn't hold but it's possible

    @test typeof(fit_cmte_pot1) <: CommitteePotential{<:PolynomialChaos}
    @test all([length(member.basis)==10 for member in fit_cmte_pot1.members])
    @test all([length(member.params)==10 for member in fit_cmte_pot1.members])
    @test length(Set([member.params for member in fit_cmte_pot1.members])) == 10 #uniqueness test again

    clp2 = deepcopy(clp_template)
    fit_cmte_pot2 = learn!(clp2, cmte_pot_template,train_systems;frac=0.7, train_subset_idxs=[i for i in 1:10])
    @test all(length.(clp2.cmte_indices) .== 7) # 0.7*10=7
    @test length(Set([indices for indices in clp2.cmte_indices])) == 10 # ensure that each index sets is unique
    @test typeof(fit_cmte_pot2) <: CommitteePotential{<:PolynomialChaos}
    @test length(Set([member.params for member in fit_cmte_pot2.members])) == 10 #uniqueness test again


    xnew = [[0.40343423993326244, 0.4226044780958752],
            [0.4173669601905878, 0.35463425253425607],
            [0.38102227188501714, 0.4353723003822222]]
    new_systems = Ensemble(ref,xnew)
    new_trainset = reduce(vcat, [train_systems, new_systems])

    fit_cmte_pot3 = learn!(clp1,fit_cmte_pot1,length(xnew),new_trainset)
    @test typeof(fit_cmte_pot3) <: CommitteePotential{<:PolynomialChaos}
    @test all([member.basis for member in fit_cmte_pot3.members] .== [member.basis for member in fit_cmte_pot1.members])
    @test all([member.params for member in fit_cmte_pot3.members] .!= [member.params for member in fit_cmte_pot1.members])

    #check that it properly appended new configs to all cmte indices
    @test all([indices[end-2:end] for indices in clp1.cmte_indices] .== Ref([21,22,23]))

end