using Cairn
using PotentialLearning: learn!
using Molly
using AtomisticQoIs
using SpecialPolynomials: Jacobi
using Statistics
using Test
using StaticArrays

pce_template = PolynomialChaos(3, 2, Jacobi{0.5,0.5})
ref = MullerBrownRot()

pce_param_set1 = [[48.7321 95.9563 11.8879 3.56038 94.8206 51.6035 69.2488 37.2358 81.8482 40.592];
                  [81.6018 30.8123 56.9185 98.2884 17.8753 2.37024 65.9343 12.8787 74.2257 12.8148];
                  [6.65932 62.8262 0.61548 91.5065 76.1619 27.3688 91.6777 66.1589 34.1041 8.07007];
                  [55.9341 15.572  19.6692 95.2555 5.81737 20.1318 68.1779 88.3704 6.86962 30.4034];]

cmte_pot1 = CommitteePotential([let
                                    pce = deepcopy(pce_template)
                                    pce.params = pce_param_set1[i,:]
                                    pce
                                end for i in 1:4],1;
                                energy_units=u"kJ*mol^-1",
                                length_units=u"nm")

pce_higher_order = PolynomialChaos(6,2,Jacobi{0.5,0.5})
# This was trained on the below trainset
higher_order_params = [67907.56073774067, -236495.6277340431, -94438.21720006624, -1.1027943198011185e6, 2.601195430270954e6,
                       -1.4384145196517883e6, 856453.3926645768, -1.919090189371708e6, 209838.17630944552, 1.7330525459233276e6,
                       -144247.711526727, -462931.1400117123, 2.7121356797566977e6, -4.7839183313770555e6, 293227.31719337555,
                       288680.535993174, 248511.9914310756, -1.2393942753533467e6, 3.10558574258223e6, 1.0398585035789298e6,
                       -362854.10291637527, -203146.76785191864, -168901.16206023595,515492.57634836657, -1.4625786324220416e6,
                       -792892.9937413529, 298007.63514007855,-17034.1204029709]

pce_higher_order.params = higher_order_params
trainset = [[0.4519952180711433, 0.5497448756263469],
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

trainset = Ensemble(ref,trainset)

xcoords1 = [0.096079218176701, -0.9623723102484034]
xcoords2 = [0.096079218176701, 0.9623723102484034]
xcoords3 = [-1.0,1.0]
xcoords4 = [2.0,1.0]
base_sys1 = System(ref,xcoords1)

main_msys = System(base_sys1;
                   general_inters = (pce_higher_order,),
                   loggers = (coords=CoordinateLogger(1;dims=2),
                              pe=PotentialEnergyLogger(1)),
                    data = Dict(:example_field => 1.0)
                   )

mean_energy = CmteEnergy(Statistics.mean; strip_units=false)

mean_std_fcomp = CmteFlatForces((cmte=Statistics.std,
                                 coord_and_atom=Statistics.mean); strip_units=false)

@testset "GreedySelector tests" begin
    ss_test_al = ALRoutine(trainset=trainset)
    gs = GreedySelector()

    main_msys_copy1 = deepcopy(main_msys)

    @test length(ss_test_al.trainset) == 20
    updated_trainset, new_configs = update_trainset!(gs,main_msys_copy1,ss_test_al)

    @test length(updated_trainset) == 21
    @test length(new_configs) == 1
    @test updated_trainset[end] == new_configs[1]

    # Ensuring that the system is what was expected and that the system is stripped
    @test all(new_configs[1].coords .== Ref(SVector{2}(xcoords1*u"nm")))
    @test new_configs[1].boundary == main_msys_copy1.boundary
    @test new_configs[1].general_inters == ()
    @test new_configs[1].loggers == ()
    @test isnothing(new_configs[1].data)
end

@testset "DefaultALDataSpec and Simple2DPotErrors tests" begin

    trigg1 = CmteTrigger(mean_energy,>,100.0u"kJ*mol^-1";
                        logger_spec=(:cmte_mean_energy, 1))
    trigg2 = CmteTrigger(mean_std_fcomp, >, 200.0u"kJ*mol^-1*nm^-1";
                        logger_spec=(:mean_std_fcomp,2))
    sct = SharedCmteTrigger((trigg1,trigg2),cmte_pot1)

    main_msys_copy2 = deepcopy(main_msys)
    dataspec_test_sys = initialize_triggers((sct,),main_msys_copy2)

    coords_eval = potential_grid_2d(ref,[[-4.4,1.5],[-2,2]],0.04,cutoff=800)
    sys_eval = Ensemble(ref,coords_eval)
    ζ = [ustrip.(Vector(coords)) for coords in coords_eval]
    GQint = GaussQuadrature(ζ,ones(length(ζ)) ./length(ζ))
    error_spec = Simple2DPotErrors(GQint,true)

    dataspec = DefaultALDataSpec(error_spec,true,true,true,true)

    dataspec_test_al = ALRoutine(;ref=ref,
                                  mlip=pce_higher_order,
                                  triggers=(sct,),
                                  trainset = trainset,
                                  ss=GreedySelector(),
                                  aldata_spec = dataspec
                                )

    al_data = initialize_al_record(dataspec,nothing,dataspec_test_al)
    @test al_data["trigger_steps"] == []
    @test al_data["initial_mlip_params"] == higher_order_params
    @test al_data["mlip_params"] == []
    @test al_data["error_hist"]["rmse_e"] == []
    @test al_data["error_hist"]["rmse_f"] == []
    @test al_data["error_hist"]["fd"] == []
    @test al_data["activated_trigger_logs"][:cmte_mean_energy] == []
    @test al_data["activated_trigger_logs"][:mean_std_fcomp] == []

    dataspec_test_al.cache[:step_n] = 8
    trigger_activated!(sct,dataspec_test_sys,dataspec_test_al) # to call log_property
    updated_trainset, dataspec_test_al.cache[:trainset_changes] = update_trainset!(dataspec_test_al.ss,
                                                                                   dataspec_test_sys,
                                                                                   dataspec_test_al)


    record_al_record!(dataspec, al_data, dataspec_test_sys, dataspec_test_al)
    @test al_data["trigger_steps"] == [8,]

    #TODO test this better after the proper retrain!()
    @test length(al_data["mlip_params"]) == 1
    @test al_data["mlip_params"][1] == higher_order_params

    @test all(al_data["error_hist"]["rmse_e"] .≈ [3.3089711922358556e9])
    @test all(al_data["error_hist"]["rmse_f"] .≈ [4.665834433478545e9])
    @test all(al_data["error_hist"]["fd"] .≈ [5.124167107785014e16])

    @test compute(mean_energy, base_sys1, cmte_pot1) ≈ 26.946218132558347u"kJ*mol^-1"
    @test compute(mean_std_fcomp, base_sys1,cmte_pot1) ≈ 177.85936562044168u"kJ*mol^-1*nm^-1"

    @test all(al_data["activated_trigger_logs"][:cmte_mean_energy] .≈ [26.946218132558347]*u"kJ*mol^-1")
    @test all(al_data["activated_trigger_logs"][:mean_std_fcomp] .≈ [177.85936562044168]*u"kJ*mol^-1*nm^-1")
end

@testset "retrain!() tests" begin
    #### reference fit for standard PCE
    ilp1 = InefficientLearningProblem(;ref=ref)
    ref_pce = deepcopy(pce_template)

    ref_fitted_pce = learn(ilp1,ref_pce,trainset)
    @test ref_fitted_pce.params ≈ [-335.3750515214229,
                                    615.4631933197177,
                                    247.61997737909118,
                                    -351.787767444939,
                                    -976.3218308181512,
                                    342.07362979908646,
                                    -804.7445886595718,
                                    2167.8887229673587,
                                    -1363.034451052221,
                                    539.6457680262732]

    #### retrain!() test with standard PCE
    retrain_test_al1 = ALRoutine(;ref=ref,
                                  mlip=deepcopy(pce_template),
                                  trainset=trainset,
                                  lp=InefficientLearningProblem(;ref=ref))

    main_msys_copy3 = deepcopy(main_msys)
    fitted_pce = retrain!(retrain_test_al1.lp, main_msys_copy3, retrain_test_al1)
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


    #### initializing relevant cmte_pot
    cmte_pot_template = CommitteePotential([deepcopy(pce_template) for _ in 1:4], 1)

    ref_clp_template = SubsampleAppendCmteRetrain(InefficientLearningProblem(;ref=ref),
                                     [[4,13,3,19,11,14,5,8,15,12,16,9,18,17],  # 14 indices per committee
                                     [8,11,12,20,16,18,4,3,1,9,10,14,6,7],
                                     [17,14,18,9,11,16,12,3,8,20,13,19,6,7],
                                     [17,1,4,16,9,2,15,7,20,10,6,5,14,19]])

    ref_cmte_pot_template = learn(deepcopy(ref_clp_template),cmte_pot_template,trainset) #shouldn't modify the clp, but I'm copying just in case
    ref_clp = deepcopy(ref_clp_template)

    check_cmte_pot = deepcopy(ref_cmte_pot_template)
    check_clp = deepcopy(ref_clp_template)

    ### reference committee potential train with new configs

    xnew = [[0.40343423993326244, 0.4226044780958752],
            [0.4173669601905878, 0.35463425253425607],
            [0.38102227188501714, 0.4353723003822222]]
    new_systems = Ensemble(ref,xnew)
    new_trainset = reduce(vcat, [trainset, new_systems])

    ref_updated_cmte_pot = learn!(ref_clp,ref_cmte_pot_template,length(new_systems),new_trainset)
    @test all([indices[end-2:end] for indices in ref_clp.cmte_indices] .== Ref([21,22,23]))
    # as always, these tests can be modified to if this is a bit of a fragile approach
    @test all([member.params for member in ref_updated_cmte_pot.members] .≈ [[-203.08410824815675, 816.7427491527678, -460.2407047701491, -579.8590258737155, -78.88841605027702,
                                                                            479.09854489024167, -295.464884104602, 1032.015001081525, -1106.9150704740443, 393.7335304371087],
                                                                           [-170.10514324571412, 663.3863637588346, -323.08457833791846, -625.1906994221896, 80.96559584975009,
                                                                            321.3751699104648, -288.4352742841502, 933.611430034781, -985.6173209444495, 338.84477894997883],
                                                                           [-345.81438206263584, 836.194934487695, 3.7216046749122524, -1395.0108315072628, 354.7054366264242,
                                                                            -201.4098287401222, -533.9467954649637, 1976.965230521249, -1512.687307437898, 721.8023707778311],
                                                                           [16.252835550456563, -1040.1012178913475, 1451.4744245744116, 1243.3755712378018, -1747.732512005455,
                                                                            -335.33576592976954, -1623.369161716008, 2379.6773343675736, -564.6135281491161, 375.1589206509789]])

    ### retrain committee potential test
    retrain_test_al2 = ALRoutine(;ref=ref,
                                  mlip=check_cmte_pot,
                                  trainset=new_trainset,
                                  lp=check_clp)
    retrain_test_al2.cache[:trainset_changes] = new_systems

    check_updated_cmte_pot = retrain!(retrain_test_al2.lp, main_msys_copy3, retrain_test_al2)
    @test all([indices[end-2:end] for indices in check_clp.cmte_indices] .== Ref([21,22,23]))
    @test all([member.params for member in check_updated_cmte_pot.members] .≈ [[-203.08410824815675, 816.7427491527678, -460.2407047701491, -579.8590258737155, -78.88841605027702,
                                                                            479.09854489024167, -295.464884104602, 1032.015001081525, -1106.9150704740443, 393.7335304371087],
                                                                           [-170.10514324571412, 663.3863637588346, -323.08457833791846, -625.1906994221896, 80.96559584975009,
                                                                            321.3751699104648, -288.4352742841502, 933.611430034781, -985.6173209444495, 338.84477894997883],
                                                                           [-345.81438206263584, 836.194934487695, 3.7216046749122524, -1395.0108315072628, 354.7054366264242,
                                                                            -201.4098287401222, -533.9467954649637, 1976.965230521249, -1512.687307437898, 721.8023707778311],
                                                                           [16.252835550456563, -1040.1012178913475, 1451.4744245744116, 1243.3755712378018, -1747.732512005455,
                                                                            -335.33576592976954, -1623.369161716008, 2379.6773343675736, -564.6135281491161, 375.1589206509789]])

    ### update_trigger test with CmteTrigger
    check_cmte_trigger = CmteTrigger(mean_energy,>,100.0u"kJ*mol^-1";
                                     cmte_pot = deepcopy(ref_cmte_pot_template),
                                     logger_spec=(:cmte_mean_energy, 1))

    update_test_al1 = ALRoutine(;trainset=new_trainset,
                                 triggers=(check_cmte_trigger,),
                                 trigger_updates=(deepcopy(ref_clp_template),)
                                )
    update_test_al1.cache[:trainset_changes] = new_systems

    new_triggers1 = update_triggers!(update_test_al1.triggers,
                                     update_test_al1.trigger_updates,
                                     main_msys_copy3,
                                     update_test_al1)

    @test new_triggers1[1].cmte_qoi == mean_energy
    @test new_triggers1[1].compare == >
    @test new_triggers1[1].thresh == 100.0u"kJ*mol^-1"
    @test all([member.params for member in new_triggers1[1].cmte_pot.members] .≈ [[-203.08410824815675, 816.7427491527678, -460.2407047701491, -579.8590258737155, -78.88841605027702,
                                                                                 479.09854489024167, -295.464884104602, 1032.015001081525, -1106.9150704740443, 393.7335304371087],
                                                                                [-170.10514324571412, 663.3863637588346, -323.08457833791846, -625.1906994221896, 80.96559584975009,
                                                                                 321.3751699104648, -288.4352742841502, 933.611430034781, -985.6173209444495, 338.84477894997883],
                                                                                [-345.81438206263584, 836.194934487695, 3.7216046749122524, -1395.0108315072628, 354.7054366264242,
                                                                                 -201.4098287401222, -533.9467954649637, 1976.965230521249, -1512.687307437898, 721.8023707778311],
                                                                                [16.252835550456563, -1040.1012178913475, 1451.4744245744116, 1243.3755712378018, -1747.732512005455,
                                                                                 -335.33576592976954, -1623.369161716008, 2379.6773343675736, -564.6135281491161, 375.1589206509789]])

    @test all([indices[end-2:end] for indices in update_test_al1.trigger_updates[1].cmte_indices] .== Ref([21,22,23]))

    #### update_trigger test with SharedCmteTrigger
    subtrigger1 = CmteTrigger(mean_energy,>,100.0u"kJ*mol^-1";
                              logger_spec=(:cmte_mean_energy, 1))
    subtrigger2 = CmteTrigger(mean_std_fcomp, >, 200.0u"kJ*mol^-1*nm^-1";
                             logger_spec=(:mean_std_fcomp,2))

    check_sct_trigger = SharedCmteTrigger((subtrigger1,subtrigger2),deepcopy(ref_cmte_pot_template))

    update_test_al2 = ALRoutine(;trainset=new_trainset,
                                 triggers=(check_sct_trigger,),
                                 trigger_updates=(deepcopy(ref_clp_template),))

    update_test_al2.cache[:trainset_changes] = new_systems

    new_triggers2 = update_triggers!(update_test_al2.triggers,
                                     update_test_al2.trigger_updates,
                                     main_msys_copy3,
                                     update_test_al2)

    @test new_triggers2[1].subtriggers[1].cmte_qoi == mean_energy
    @test new_triggers2[1].subtriggers[1].compare == >
    @test new_triggers2[1].subtriggers[1].thresh == 100.0u"kJ*mol^-1"

    @test new_triggers2[1].subtriggers[2].cmte_qoi == mean_std_fcomp
    @test new_triggers2[1].subtriggers[2].compare == >
    @test new_triggers2[1].subtriggers[2].thresh == 200.0u"kJ*mol^-1*nm^-1"

    @test all([member.params for member in new_triggers2[1].cmte_pot.members] .≈ [[-203.08410824815675, 816.7427491527678, -460.2407047701491, -579.8590258737155, -78.88841605027702,
                                                                                 479.09854489024167, -295.464884104602, 1032.015001081525, -1106.9150704740443, 393.7335304371087],
                                                                                [-170.10514324571412, 663.3863637588346, -323.08457833791846, -625.1906994221896, 80.96559584975009,
                                                                                 321.3751699104648, -288.4352742841502, 933.611430034781, -985.6173209444495, 338.84477894997883],
                                                                                [-345.81438206263584, 836.194934487695, 3.7216046749122524, -1395.0108315072628, 354.7054366264242,
                                                                                 -201.4098287401222, -533.9467954649637, 1976.965230521249, -1512.687307437898, 721.8023707778311],
                                                                                [16.252835550456563, -1040.1012178913475, 1451.4744245744116, 1243.3755712378018, -1747.732512005455,
                                                                                 -335.33576592976954, -1623.369161716008, 2379.6773343675736, -564.6135281491161, 375.1589206509789]])


    @test all([indices[end-2:end] for indices in update_test_al2.trigger_updates[1].cmte_indices] .== Ref([21,22,23]))

end