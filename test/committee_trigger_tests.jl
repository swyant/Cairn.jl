using Cairn
using Molly
using Statistics
using SpecialPolynomials: Jacobi
using LinearAlgebra: norm
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

pce_param_set2 = [[184.49 -1721.549 1788.434 2440.284 -2570.344 -114.643 -2474.031 3084.844 -799.532 414.281];
                  [-510.463 440.639 964.72 -428.42 -1526.669 24.677 -1077.07 2971.987 -1407.748 649.078];
                  [-635.528      942.565 761.209 -1431.472 -813.293 -256.832 -1201.521 3837.554 -2240.273 1090.312];
                  [-434.544 987.383 93.775 -972.452 -604.285 205.344 -1203.359 3306.645 -2327.126 1003.494]]

cmte_pot2 = CommitteePotential([let
                                    pce = deepcopy(pce_template)
                                    pce.params = pce_param_set2[i,:]
                                    pce
                                end for i in 1:4],1;
                                energy_units=u"kJ*mol^-1",
                                length_units=u"nm")

pce_param_set3 = [[58.06549 60.73548 -179.64167 39.72457 76.47814 79.88366 -79.71651 -46.46783 -87.48965 55.40965];
                  [84.66007 26.98498 -206.08551 133.13805 48.15514 124.42591 -20.2109 -224.7014 26.43595 -7.96472];
                  [67.89833 32.52971 -166.66804 69.36616 69.78022 69.55492 -94.50418 -44.28919 -78.3022 55.26448];
                  [39.35999 41.74056 -82.0616 17.31019 6.25778 30.81547 -0.06884 -122.45153 44.97271 12.49682]]

cmte_pot3 = CommitteePotential([let
                                    pce = deepcopy(pce_template)
                                    pce.params = pce_param_set3[i,:]
                                    pce
                                end for i in 1:4],1;
                                energy_units=u"kJ*mol^-1",
                                length_units=u"nm")

xcoords1 = [0.096079218176701, -0.9623723102484034]
xcoords2 = [0.096079218176701, 0.9623723102484034]
xcoords3 = [-1.0,1.0]
xcoords4 = [2.0,1.0]
base_sys1 = System(ref,xcoords1)
base_sys2 = System(ref,xcoords2)
base_sys3 = System(ref,xcoords3)
base_sys4 = System(ref,xcoords4)

mean_energy = CmteEnergy(Statistics.mean; strip_units=false)
mean_energy_stripped = CmteEnergy(Statistics.mean; strip_units=true)

mean_std_fcomp = CmteFlatForces((cmte=Statistics.std,
                                 coord_and_atom=Statistics.mean); strip_units=false)

function num_above(array, thresh)
   return count(array .> thresh)
end

function all_below(array, thresh)
   return all(array .< thresh)
end

# how many committees predict all force components with absolute values below 400
num_cmte_large_forces = CmteFlatForces((coord_and_atom = (x -> all_below(abs.(x),400)),
                                        cmte           = count);
                                        strip_units = true
                                      )

all_cmte_large_forces = CmteFlatForces((coord_and_atom = (x -> all_below(abs.(x),400)),
                                        cmte           = all);
                                        strip_units = true
                                      )

# how many committees predict at least one force component with an absolute value above 1000
num_cmte_max_force = CmteFlatForces((coord_and_atom = (x -> maximum(abs.(x))   ),
                                     cmte           = (x -> num_above(x,1000) ));
                                     strip_units = true
                                    )

struct DummyALRoutine
    cache::Dict
    function DummyALRoutine()
        cache_dict = Dict()
        dummy_alroutine = new(cache_dict)
        dummy_alroutine
    end
end

@testset "Basic SimpleTriggerLogger Tests" begin
    simple_logger1 = SimpleTriggerLogger()
    @test typeof(simple_logger1) <: SimpleTriggerLogger{Float64}
    @test eltype(simple_logger1.history) <: Float64
    @test simple_logger1.n_steps == 0
    @test isnothing(simple_logger1.observable)

    @test_throws MethodError log_property!(simple_logger1,10,1)
    @test_throws MethodError log_property!(simple_logger1,Float32(10.0),1)
    log_property!(simple_logger1, 10.0, 1)
    @test simple_logger1.observable == 10.0
    @test length(simple_logger1.history) == 0
    log_property!(simple_logger1, 15.0, 1)
    @test simple_logger1.observable == 15.0
    @test length(simple_logger1.history) == 0

    Cairn.reset_observable!(simple_logger1)
    @test isnothing(simple_logger1.observable)

    simple_logger2 = SimpleTriggerLogger(1)
    @test typeof(simple_logger2) <: SimpleTriggerLogger{Float64}
    @test eltype(simple_logger2.history) <: Float64
    @test simple_logger2.n_steps == 1
    @test isnothing(simple_logger2.observable)

    log_property!(simple_logger2,1.0,1)
    @test simple_logger2.observable == 1.0
    @test simple_logger2.history == [1.0]
    log_property!(simple_logger2,3.0,1)
    @test simple_logger2.observable == 3.0
    @test simple_logger2.history == [1.0,3.0]

    Cairn.reset_observable!(simple_logger2)
    @test isnothing(simple_logger2.observable)
    @test simple_logger2.history == [1.0,3.0]

    simple_logger3 = SimpleTriggerLogger(3, Int64)
    @test typeof(simple_logger3) <: SimpleTriggerLogger{Int64}
    @test eltype(simple_logger3.history) <: Int64
    @test simple_logger3.n_steps == 3
    @test isnothing(simple_logger3.observable)

    @test_throws MethodError log_property!(simple_logger3,1.0,1)
    log_property!(simple_logger3,-1,1)
    @test simple_logger3.observable == -1
    @test length(simple_logger3.history) == 0

    for i in 1:6
        log_property!(simple_logger3,i,i)
    end
    @test simple_logger3.observable == 6
    @test simple_logger3.history == [3,6]


    # run_loggers should not do anything
    dummy_simple_logger = SimpleTriggerLogger(1,Float64)
    dummy_sys = System(base_sys1;
                       loggers = (coords=CoordinateLogger(1; dims=2),
                                  pe=PotentialEnergyLogger(1),
                                  stl=dummy_simple_logger))
    run_loggers!(dummy_sys,nothing,1)
    @test length(dummy_sys.loggers.coords.history) == 1
    @test length(dummy_sys.loggers.pe.history) == 1
    @test length(dummy_sys.loggers.stl.history) == 0
end


@testset "CmteTrigger Tests" begin
    trigger1 = CmteTrigger(mean_energy,>,0.0u"kJ*mol^-1")
    @test typeof(trigger1) <: CmteTrigger{CmteEnergy,typeof(>),typeof(1.0u"kJ*mol^-1")}
    @test isnothing(trigger1.cmte_pot)
    @test isnothing(trigger1.logger_spec)

    trigger2 = CmteTrigger(mean_energy_stripped,>, 0.0)
    @test typeof(trigger2) <:CmteTrigger{CmteEnergy,typeof(>),Float64}

    trigger3 = CmteTrigger(mean_std_fcomp,>,1000.0u"kJ*nm^-1*mol^-1")
    @test typeof(trigger3) <: CmteTrigger{CmteFlatForces,typeof(>),typeof(1.0u"kJ*nm^-1*mol^-1")}

    trigger4 = CmteTrigger(mean_energy, >, 0.0u"kJ*mol^-1";
                           cmte_pot=cmte_pot1,
                           logger_spec=(:cmte_mean_energy,2) )

    @test trigger4.cmte_pot == cmte_pot1
    @test trigger4.logger_spec == (:cmte_mean_energy,2)

    @test_throws "Threshold in CmteTrigger needs to be a float, integer," CmteTrigger(mean_energy,>,"hey")
    @test_throws "Threshold in CmteTrigger needs to be a float, integer," CmteTrigger(mean_energy,>,5+6im)
    @test_throws "logger_spec needs to be Tuple{Symbol,Int64}" CmteTrigger(mean_energy_stripped,>,0.0;
                                                                           logger_spec = ("mean_energy",1))
    @test_throws "logger_spec needs to be Tuple{Symbol,Int64}" CmteTrigger(mean_energy_stripped,>,0.0;
                                                                           logger_spec = (:mean_energy,1.0))


    msys_nologgers = deepcopy(base_sys1)
    msys_existing_loggers = System(base_sys1;
                                   loggers = (coord=CoordinateLogger(1),
                                              pe=PotentialEnergyLogger(1)))

    # initialize_triggers will always force system loggers to be a NamedTuple, even if logger_spec is nothing
    msys_nologgers_trigg1 = initialize_triggers((trigger1,),msys_nologgers)
    @test typeof(msys_nologgers_trigg1.loggers) <: NamedTuple
    @test length(msys_nologgers_trigg1.loggers) == 0

    # check loggers are appropriately appended when starting system has no loggers
    msys_nologgers_trigg4 = initialize_triggers((trigger4,),msys_nologgers)
    @test keys(msys_nologgers_trigg4.loggers) == (:cmte_mean_energy,)
    @test typeof(msys_nologgers_trigg4.loggers.cmte_mean_energy) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1")}
    @test msys_nologgers_trigg4.loggers.cmte_mean_energy.n_steps == 2

    # check that existing loggers are unchanged if trigger has no logger spec
    msys_existing_loggers_trigg1 = initialize_triggers((trigger1,),msys_existing_loggers)
    @test msys_existing_loggers_trigg1.loggers == msys_existing_loggers.loggers

    # check that new logger is correctly appended to existing loggers
    msys_existing_loggers_trigg4 = initialize_triggers((trigger4,),msys_existing_loggers)
    @test typeof(msys_existing_loggers_trigg4.loggers) <: NamedTuple
    @test keys(msys_existing_loggers_trigg4.loggers) == (:coord, :pe, :cmte_mean_energy)
    @test typeof(msys_existing_loggers_trigg4.loggers.cmte_mean_energy) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1")}
    @test msys_existing_loggers_trigg4.loggers.cmte_mean_energy.n_steps == 2

    # checking situations that produce errors when trying to initialize_triggers
    msys_plain_tuple_loggers = System(base_sys1;
                                      loggers = (CoordinateLogger(1)))
    msys_field_exists = System(base_sys1;
                               loggers = (cmte_mean_energy=SimpleTriggerLogger(1),))

    #I should probably handle this case in the future
    @test_throws "Can't handle a case where sys.loggers is Tuple with finite size" bad_sys = initialize_triggers((trigger1,),msys_plain_tuple_loggers)
    @test_throws "Symbol provided in trigger.logger_spec is already used in the System.loggers"  bad_sys = initialize_triggers((trigger4,),msys_field_exists)


    main_sys = System(base_sys1;
                       loggers = (pe=PotentialEnergyLogger(1),))

    dummy_alroutine = DummyALRoutine()
    dummy_alroutine.cache[:step_n] = 1

    # checking trigger comparing a Unitful.quantity returns true/false when appropriate, checking that SimpleTriggerLogger is appropriately modified
    mean_e_trigger = CmteTrigger(mean_energy, >, 100.0u"kJ*mol^-1";
                                 cmte_pot = cmte_pot1,
                                 logger_spec = (:cmte_mean_energy, 1))
    msys1_w_trigg = initialize_triggers((mean_e_trigger,),main_sys)
    @test compute(mean_energy,base_sys1,cmte_pot1) ≈ 26.946218132558347u"kJ*mol^-1"
    @test trigger_activated!(mean_e_trigger, msys1_w_trigg, dummy_alroutine) == false
    @test msys1_w_trigg.loggers[:cmte_mean_energy].observable ≈ 26.946218132558347u"kJ*mol^-1"
    msys1_w_trigg.coords = [SVector{2}(xcoords2*u"nm")] # same coords as base_sys2
    @test compute(mean_energy,base_sys2,cmte_pot1) ≈ 238.62856102151517u"kJ*mol^-1"
    dummy_alroutine.cache[:step_n] = 2
    @test trigger_activated!(mean_e_trigger, msys1_w_trigg, dummy_alroutine) == true
    @test msys1_w_trigg.loggers[:cmte_mean_energy].observable ≈ 238.62856102151517u"kJ*mol^-1"
    @test msys1_w_trigg.loggers[:cmte_mean_energy].history ≈ [26.946218132558347,238.62856102151517]u"kJ*mol^-1"
    @test length(msys1_w_trigg.loggers[:pe].history) == 0
    perstep_reset!((mean_e_trigger,),msys1_w_trigg)
    @test isnothing(msys1_w_trigg.loggers[:cmte_mean_energy].observable)

    # checking that float comparisons work, and that SimpleTriggerLogger records history appropriately
    mean_s_e_trigger = CmteTrigger(mean_energy_stripped, >, 100.0;
                                  cmte_pot = cmte_pot1,
                                  logger_spec = (:cmte_mean_energy, 2))

    dummy_alroutine.cache[:step_n] = 1
    msys2_w_trigg = initialize_triggers((mean_s_e_trigger,),main_sys)
    @test compute(mean_energy_stripped,base_sys1,cmte_pot1) ≈ 26.946218132558347
    @test trigger_activated!(mean_s_e_trigger, msys2_w_trigg, dummy_alroutine) == false
    @test msys2_w_trigg.loggers[:cmte_mean_energy].observable ≈ 26.946218132558347
    msys2_w_trigg.coords = [SVector{2}(xcoords2*u"nm")]
    @test compute(mean_energy_stripped,base_sys2,cmte_pot1) ≈ 238.62856102151517
    dummy_alroutine.cache[:step_n] = 2
    @test trigger_activated!(mean_s_e_trigger, msys2_w_trigg, dummy_alroutine) == true
    @test msys2_w_trigg.loggers[:cmte_mean_energy].observable ≈ 238.62856102151517
    @test msys2_w_trigg.loggers[:cmte_mean_energy].history ≈ [238.62856102151517,]

    # check that integer comparison works, also lt operator instead of gt operator. No loggers
    num_cmte_trigger = CmteTrigger(num_cmte_large_forces, <, 4;
                                   cmte_pot = cmte_pot1)

    dummy_alroutine.cache[:step_n] = 1
    msys3_w_trigg = initialize_triggers((num_cmte_trigger,),main_sys)
    @test compute(num_cmte_large_forces,base_sys1,cmte_pot1) == 4
    @test trigger_activated!(num_cmte_trigger,msys3_w_trigg,dummy_alroutine) == false
    msys3_w_trigg.coords = [SVector{2}(xcoords2*u"nm")]
    @test compute(num_cmte_large_forces,base_sys2,cmte_pot1) == 1
    dummy_alroutine.cache[:step_n] = 2
    @test trigger_activated!(num_cmte_trigger,msys3_w_trigg,dummy_alroutine) == true

    # check a boolean comparison, == operator
    all_cmte_trigger = CmteTrigger(all_cmte_large_forces,==,false;
                                   cmte_pot = cmte_pot1)

    dummy_alroutine.cache[:step_n] = 1
    msys4_w_trigg = initialize_triggers((all_cmte_trigger,),main_sys)
    @test compute(all_cmte_large_forces,base_sys1,cmte_pot1) == true
    @test trigger_activated!(all_cmte_trigger,msys4_w_trigg,dummy_alroutine) == false
    msys4_w_trigg.coords = [SVector{2}(xcoords2*u"nm")]
    @test compute(all_cmte_large_forces,base_sys2,cmte_pot1) == false
    dummy_alroutine.cache[:step_n] = 2
    @test trigger_activated!(all_cmte_trigger,msys4_w_trigg,dummy_alroutine) == true

    dummy_alroutine.cache[:step_n] = 1

    # User must ensure that return time of cmte_qoi is the same as the threshold
    trigger_bad = CmteTrigger(mean_energy,<,0.0; # mean_energy returns Unitful.Quantity, but threshold is Float64
                              cmte_pot = cmte_pot1)
    msys_bad = initialize_triggers((trigger_bad,),main_sys)
    @test_throws "return type of compute(cmte_qoi,...) is not the same type" trigger_activated!(trigger_bad,msys_bad,dummy_alroutine)

    # Testing that the appropriate cmte_pot is used.
    # If trigger has cmte_pot, that will take precedence, otherwise fall back to general_inters cmte_pot
    # If neither are available, throw error
    sys_w_cmte = System(base_sys1;
                        general_inters = (cmte_pot2,))
    alt_mean_e_trigger1 = CmteTrigger(mean_energy,<,0.0u"kJ*mol^-1";
                                     cmte_pot = cmte_pot1)
    alt_mean_e_trigger2 = CmteTrigger(mean_energy,<,0.0u"kJ*mol^-1")

    @test compute(mean_energy,base_sys1,cmte_pot1) ≈ 26.946218132558347u"kJ*mol^-1" # mostly writing this for reference
    @test compute(mean_energy,base_sys1,cmte_pot2) ≈ -9510.118425779501u"kJ*mol^-1"

    # don't really need to initialize_trigger here but that's the normal process so just in case
    sys_w_cmte_1 = initialize_triggers((alt_mean_e_trigger1,),sys_w_cmte)
    @test trigger_activated!(alt_mean_e_trigger1,sys_w_cmte_1,dummy_alroutine) == false # because value should be 26.9 using cmte_pot1
    sys_w_cmte_2 = initialize_triggers((alt_mean_e_trigger2,),sys_w_cmte)
    @test trigger_activated!(alt_mean_e_trigger2,sys_w_cmte_2,dummy_alroutine) == true # because value should be -9510, using cmte_pot2

    # neither trigger nor general_inters has committee potential, so should throw error
    sys_wo_cmte = initialize_triggers((alt_mean_e_trigger2,),main_sys)
    @test_throws "No committee potential available for trigger activation" trigger_activated!(alt_mean_e_trigger2, sys_wo_cmte,dummy_alroutine)

    # What if you want to log an intermediate trigger-related QoI. Don't provide an easy way for that...
    # In theory could register a standard logger with the intermediate QoI, and could add a cache field to the regular cmte_trigger
end

@testset "SharedCmteTrigger Tests" begin

    #### basic constructor tests
    trigg1 = CmteTrigger(mean_energy,>,100.0u"kJ*mol^-1";
                         logger_spec=(:cmte_mean_energy, 1))
    trigg2 = CmteTrigger(mean_std_fcomp, >, 200.0u"kJ*mol^-1*nm^-1";
                         logger_spec=(:mean_std_fcomp,2))
    trigg3 = CmteTrigger(num_cmte_max_force, >, 1)

    # testing inner constructor with all defaults
    sct1 = SharedCmteTrigger((trigg1,trigg2,trigg3))
    @test typeof(sct1) <: SharedCmteTrigger
    @test isnothing(sct1.cmte_pot)
    @test isnothing(sct1.energy_cache_field)
    @test isnothing(sct1.force_cache_field)

    # testing inner constructor with defaults except for cmte_pot
    sct2 = SharedCmteTrigger((trigg1,trigg2,trigg3), cmte_pot1)
    @test typeof(sct2) <: SharedCmteTrigger
    @test sct2.cmte_pot == cmte_pot1
    @test isnothing(sct2.energy_cache_field)
    @test isnothing(sct2.force_cache_field)

    # testing constructor with keywords
    sct3 = SharedCmteTrigger((trigg1,trigg2,trigg3), cmte_pot1;
                             energy_cache_field=:cmte_energies,
                             force_cache_field=:cmte_forces)
    @test typeof(sct3) <: SharedCmteTrigger
    @test sct3.cmte_pot == cmte_pot1
    @test sct3.energy_cache_field == :cmte_energies
    @test sct3.force_cache_field == :cmte_forces

    #### initialize_triggers: append_loggers tests
    msys_nologgers = deepcopy(base_sys1)
    msys_existing_loggers = System(base_sys1;
                                   loggers = (coord=CoordinateLogger(1),
                                              pe=PotentialEnergyLogger(1)))

    # loggers correctly set up when no loggers exists to start
    msys_startwnologgers = initialize_triggers((sct2,), msys_nologgers)
    @test keys(msys_startwnologgers.loggers) == (:cmte_mean_energy, :mean_std_fcomp)
    @test typeof(msys_startwnologgers.loggers.cmte_mean_energy) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1")}
    @test msys_startwnologgers.loggers.cmte_mean_energy.n_steps == 1
    @test typeof(msys_startwnologgers.loggers.mean_std_fcomp) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1*nm^-1")}
    @test msys_startwnologgers.loggers.mean_std_fcomp.n_steps == 2

    # loggers correctly set up when loggers already exists
    msys_startwloggers = initialize_triggers((sct2,), msys_existing_loggers)
    @test keys(msys_startwloggers.loggers) == (:coord,:pe, :cmte_mean_energy, :mean_std_fcomp)
    @test typeof(msys_startwloggers.loggers.cmte_mean_energy) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1")}
    @test msys_startwloggers.loggers.cmte_mean_energy.n_steps == 1
    @test typeof(msys_startwloggers.loggers.mean_std_fcomp) <: SimpleTriggerLogger{typeof(1.0u"kJ*mol^-1*nm^-1")}
    @test msys_startwloggers.loggers.mean_std_fcomp.n_steps == 2

    # Second logger has the same logger_spec symbol, should throw error
    trigg_bad_spec = CmteTrigger(mean_energy_stripped,<,-500.0;
                         logger_spec=(:cmte_mean_energy, 1))
    sct_badspec = SharedCmteTrigger((trigg1,trigg2,trigg3,trigg_bad_spec),cmte_pot1)
    @test_throws "Symbol provided in trigger.logger_spec is already used in the System.loggers" initialize_triggers((sct_badspec,), msys_nologgers)

    #### initialize_triggers: initialize_data tests

    msys_nodata = deepcopy(base_sys1)
    msys_existing_data1 = System(base_sys1;
                                data=Dict("existing"=>1)) # Dict{String,Int64}
    msys_existing_data2 = System(base_sys1;
                                 data=Dict(:exist1 => 1,
                                           "exist2" => "hey")) # Dict{Any, Any}

    # starting with system with no data, is sys.data initialized correctly?
    msys_startnodata = initialize_triggers((sct3,),msys_nodata)
    @test typeof(msys_startnodata.data) <: Dict{Any,Any}
    @test length(keys(msys_startnodata.data)) == 3
    @test all(in.( (:_reset_every_step, :cmte_energies, :cmte_forces), Ref(keys(msys_startnodata.data)))) # can't assume key order
    @test msys_startnodata.data[:_reset_every_step] ==  [:cmte_energies, :cmte_forces]
    @test isnothing(msys_startnodata.data[:cmte_energies])
    @test isnothing(msys_startnodata.data[:cmte_forces])

    # Starting with a system that has a data field of type Dict{String, Int64}, does the data field get appropriately initialized
    msys_startwdata1 = initialize_triggers((sct3,),msys_existing_data1)
    @test typeof(msys_startwdata1.data) <: Dict{Any,Any}
    @test length(keys(msys_startwdata1.data)) == 4
    @test all(in.( ("existing", :_reset_every_step, :cmte_energies, :cmte_forces), Ref(keys(msys_startwdata1.data))))
    @test msys_startwdata1.data[:_reset_every_step] == [:cmte_energies, :cmte_forces]
    @test msys_startwdata1.data["existing"] == 1
    @test isnothing(msys_startwdata1.data[:cmte_energies])
    @test isnothing(msys_startwdata1.data[:cmte_forces])

    # Starting with a system that has a data field of type Dict{Any,Any}, w/ two entries
    msys_startwdata2 = initialize_triggers((sct3,),msys_existing_data2)
    @test typeof(msys_startwdata2.data) <: Dict{Any,Any}
    @test length(keys(msys_startwdata2.data)) == 5
    @test all(in.( (:exist1, "exist2", :_reset_every_step, :cmte_energies, :cmte_forces), Ref(keys(msys_startwdata2.data))))
    @test msys_startwdata2.data[:_reset_every_step] == [:cmte_energies, :cmte_forces]
    @test msys_startwdata2.data[:exist1] == 1
    @test msys_startwdata2.data["exist2"] == "hey"
    @test isnothing(msys_startwdata2.data[:cmte_energies])
    @test isnothing(msys_startwdata2.data[:cmte_forces])

    # TODO: test SharedCmteTrigger + CmteTrigger combos

    #### boolean logic and loggers working

    msys_main_init = System(base_sys1;
                            loggers = (pe=PotentialEnergyLogger(1),))
    msys_main = initialize_triggers((sct2,),msys_main_init)

    dummy_alroutine = DummyALRoutine()
    dummy_alroutine.cache[:step_n] = 1

    # first case (base_sys1), all triggers should be false. Timestep=1
    @test compute(sct2.subtriggers[1].cmte_qoi,base_sys1,cmte_pot1) ≈ 26.946218132558347u"kJ*mol^-1"
    @test compute(sct2.subtriggers[2].cmte_qoi,base_sys1,cmte_pot1) ≈ 177.85936562044168u"kJ*mol^-1*nm^-1"
    @test compute(sct2.subtriggers[3].cmte_qoi,base_sys1,cmte_pot1) == 0

    @test trigger_activated!(sct2,msys_main,dummy_alroutine) == false
    @test msys_main.loggers[:cmte_mean_energy].observable ≈ 26.946218132558347u"kJ*mol^-1"
    @test msys_main.loggers[:mean_std_fcomp].observable ≈ 177.85936562044168u"kJ*mol^-1*nm^-1"

    # second case (base_sys2), only first trigger should be true, but overall trigger_activated will be true. Timestep=2
    msys_main.coords = [SVector{2}(xcoords2*u"nm")]
    dummy_alroutine.cache[:step_n] = 2

    @test compute(sct2.subtriggers[1].cmte_qoi,base_sys2,cmte_pot1) ≈ 238.62856102151517u"kJ*mol^-1"
    @test compute(sct2.subtriggers[2].cmte_qoi,base_sys2,cmte_pot1) ≈ 192.1288782829269u"kJ*mol^-1*nm^-1"
    @test compute(sct2.subtriggers[3].cmte_qoi,base_sys2,cmte_pot1) == 0

    @test trigger_activated!(sct2,msys_main,dummy_alroutine) == true
    @test msys_main.loggers[:cmte_mean_energy].observable ≈ 238.62856102151517u"kJ*mol^-1"
    @test msys_main.loggers[:mean_std_fcomp].observable ≈ 192.1288782829269u"kJ*mol^-1*nm^-1"

    # third case (base_sys3), only second trigger should be true, but overall trigger_activated will be true. Timestep=3
    msys_main.coords = [SVector{2}(xcoords3*u"nm")]
    dummy_alroutine.cache[:step_n] = 3

    @test compute(sct2.subtriggers[1].cmte_qoi,base_sys3,cmte_pot1) ≈ 57.33819859375001u"kJ*mol^-1"
    @test compute(sct2.subtriggers[2].cmte_qoi,base_sys3,cmte_pot1) ≈ 235.51275490040797u"kJ*mol^-1*nm^-1"
    @test compute(sct2.subtriggers[3].cmte_qoi,base_sys3,cmte_pot1) == 0

    @test trigger_activated!(sct2,msys_main,dummy_alroutine) == true
    @test msys_main.loggers[:cmte_mean_energy].observable ≈ 57.33819859375001u"kJ*mol^-1"
    @test msys_main.loggers[:mean_std_fcomp].observable ≈ 235.51275490040797u"kJ*mol^-1*nm^-1"

    # fourth case (base_sys4), only second trigger should be true, but overall trigger_activated will be true. Timestep=4
    msys_main.coords = [SVector{2}(xcoords4*u"nm")]
    dummy_alroutine.cache[:step_n] = 4

    @test compute(sct2.subtriggers[1].cmte_qoi,base_sys4,cmte_pot1) ≈ 3142.4504457812504u"kJ*mol^-1"
    @test compute(sct2.subtriggers[2].cmte_qoi,base_sys4,cmte_pot1) ≈ 1122.792294408608u"kJ*mol^-1*nm^-1"
    @test compute(sct2.subtriggers[3].cmte_qoi,base_sys4,cmte_pot1) == 4

    @test trigger_activated!(sct2,msys_main,dummy_alroutine) == true
    @test msys_main.loggers[:cmte_mean_energy].observable ≈ 3142.4504457812504u"kJ*mol^-1"
    @test msys_main.loggers[:mean_std_fcomp].observable ≈ 1122.792294408608u"kJ*mol^-1*nm^-1"

    # checking that perstep_reset
    perstep_reset!((sct2,),msys_main)
    @test isnothing(msys_main.loggers[:cmte_mean_energy].observable)
    @test isnothing(msys_main.loggers[:mean_std_fcomp].observable)

    # checking overall logger histories after these four trigger_activated!() calls
    @test msys_main.loggers[:cmte_mean_energy].history ≈ [26.946218132558347,238.62856102151517,57.33819859375001,3142.4504457812504]u"kJ*mol^-1"
    @test msys_main.loggers[:mean_std_fcomp].history ≈ [192.1288782829269,1122.792294408608]u"kJ*mol^-1*nm^-1"
    @test length(msys_main.loggers[:pe].history) == 0

    #### Checking that appropriate cmte_pot is used

    # First case, System has general_inters[1]==cmte_pot2, SharedCmteTrigger also has cmte_pot==cmte_pot1. The latter takes precedence
    msys_wcmtepot_init = System(base_sys1;
                                general_inters=(cmte_pot2,))

    msys_wcmtepot_sct2 = initialize_triggers((sct2,), msys_wcmtepot_init)

    dummy_alroutine.cache[:step_n] = 1

    #see above lines for compute with sct2.subtrigger's for expected values, trigger_activated!() should return false
    @test trigger_activated!(sct2,msys_wcmtepot_sct2,dummy_alroutine) == false
    @test msys_wcmtepot_sct2.loggers[:cmte_mean_energy].observable ≈ 26.946218132558347u"kJ*mol^-1"
    @test msys_wcmtepot_sct2.loggers[:mean_std_fcomp].observable ≈ 177.85936562044168u"kJ*mol^-1*nm^-1"

    # Second case, System still has general_inters[1]==cmte_pot2, SharedCmteTrigger does *not* have cmte_pot so cmte_pot2 should be invoked
    msys_wcmtepot_sct1 = initialize_triggers((sct1,),msys_wcmtepot_init)

    @test compute(sct1.subtriggers[1].cmte_qoi,base_sys1,cmte_pot2) ≈ -9510.118425779501u"kJ*mol^-1"
    @test compute(sct1.subtriggers[2].cmte_qoi,base_sys1,cmte_pot2) ≈ 5233.7395880969u"kJ*mol^-1*nm^-1"
    @test compute(sct1.subtriggers[3].cmte_qoi,base_sys1,cmte_pot2) == 4

    @test trigger_activated!(sct1,msys_wcmtepot_sct1,dummy_alroutine) == true
    @test msys_wcmtepot_sct1.loggers[:cmte_mean_energy].observable ≈ -9510.118425779501u"kJ*mol^-1"
    @test msys_wcmtepot_sct1.loggers[:mean_std_fcomp].observable ≈ 5233.7395880969u"kJ*mol^-1*nm^-1"

    # Third case, neither System nor SharedCmteTrigger has cmte_pot. However individual CmteTrigger's do. Doesn't matter, error should be thrown
    simple_sys_init = deepcopy(base_sys1)

    trigg1_alt = CmteTrigger(mean_energy,>,100.0u"kJ*mol^-1";
                             logger_spec=(:cmte_mean_energy, 1))
    trigg2_alt = CmteTrigger(mean_std_fcomp, >, 200.0u"kJ*mol^-1*nm^-1";
                             logger_spec=(:mean_std_fcomp,2))

    sct_alt = SharedCmteTrigger((trigg1_alt,trigg2_alt))

    simple_sys = initialize_triggers((sct_alt,),simple_sys_init)
    @test_throws "SharedCmteTrigger cannot be used if neither" trigger_activated!(sct1,simple_sys,dummy_alroutine)



    #### Checking that data is actually cached, cache data is being used

    # using msys_existing_data1 and sct3 w/ cache fields.
    # after trigger_activated!(), cache fields should be populated and will persist until
    msys_cache_test = initialize_triggers((sct3,), msys_existing_data1)
    trigger_activated!(sct3,msys_cache_test,dummy_alroutine)

    @test msys_cache_test.data[:cmte_energies] ≈ [16.894988694723455,
                                                       174.92114032265883,
                                                       4.6978599790506435,
                                                       -88.72911646619953] * u"kJ*mol^-1"

    ref_cache_forces = [[131.97621459195736, -40.62578483607613],
                        [78.55028720340587, 259.2734577408197],
                        [-57.734110333280256, -12.508601878931005],
                        [-59.59455640108558, -372.1094631947155]]
    ref_cache_forces = [[SVector{2}(ref_force_arr)*u"kJ*mol^-1*nm^-1"] for ref_force_arr in ref_cache_forces]
    @test all([all(msys_cache_test.data[:cmte_forces][i] .≈ ref_cache_forces[i]) for i in eachindex(ref_cache_forces)])

    # To check that cache fields are being used, running compute on different system, get same results as base_sys1
    # only recover correct qoi after perstep_reset!
    msys_cache_test.coords = [SVector{2}(xcoords4*u"nm")]
    @test compute(mean_energy,msys_cache_test,cmte_pot1;cache_field=:cmte_energies) ≈ 26.946218132558347u"kJ*mol^-1" # result for base_sys1
    @test compute(mean_std_fcomp,msys_cache_test,cmte_pot1;cache_field=:cmte_forces) ≈ 177.85936562044168u"kJ*mol^-1*nm^-1" # result for base_sys1

    perstep_reset!((sct3,),msys_cache_test)

    @test isnothing(msys_cache_test.data[:cmte_energies])
    @test isnothing(msys_cache_test.data[:cmte_forces])
    @test compute(mean_energy,msys_cache_test,cmte_pot1;cache_field=:cmte_energies)    ≈ 3142.4504457812504u"kJ*mol^-1" # correct result
    @test compute(mean_std_fcomp,msys_cache_test,cmte_pot1;cache_field=:cmte_forces) ≈ 1122.792294408608u"kJ*mol^-1*nm^-1" # correct result



end
