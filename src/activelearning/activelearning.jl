include("aldata.jl")
include("alroutine.jl")
include("distributions.jl")
include("kernels.jl")



export
    active_learn!




"""
    active_learn!(sys::Union{System, Vector{<:System}}, sim::Simulator, n_steps::Integer, sys_train::Vector{<:System}, ref::Union{GeneralInteraction, PairwiseInteraction}, trigger::Union{Bool, ActiveLearningTrigger}; n_threads::Integer=Threads.nthreads(), run_loggers=true)

Performs online active learning by molecular dynamics simulation defined in `sim`, using the retraining criterion defined in `trigger`.

# Arguments
- `sys::Union{System, Vector{<:System}}` : a system or ensemble of systems to simulate
- `sim::Simulator` : simulator of the equations of motion
- `n_steps::Integer` : number of simulation time steps
- `sys_train::Vector{<:System}` : ensemble of systems representing the training data
- `ref::Union{GeneralInteraction, PairwiseInteraction}` : interaction for computing reference values
- `trigger::Union{Bool, ActiveLearningTrigger}` : trigger which instantiates retraining
- `n_threads::Integer=Threads.nthreads()` : number of threads
- `run_loggers=true` : Bool for running loggers

"""
function active_learn!(sys::System,
    sim,
    n_steps::Integer,
    al::ActiveLearnRoutine;
    n_threads::Integer=Threads.nthreads(),
    run_loggers=true,
    rng=Random.GLOBAL_RNG,
)
    sys, neighbors = initialize_sim!(sys; n_threads=n_threads, run_loggers=run_loggers)
    compute_error_metrics!(al)


    for step_n in 1:n_steps
        neighbors, ksd = simulation_step!(sys,
                            sim,
                            step_n,
                            n_threads=n_threads,
                            neighbors=neighbors,
                            run_loggers=run_loggers,
                            rng=rng,
        )

        sys_new = remove_loggers(sys)
        sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new])

        # online active learning
        if trigger_activated(sys, al.triggers; sys_train=sys_train, step_n=step_n)
            al.sys_train = add_train_data()
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end



# single trajectory MD
function active_learn!(sys::System,
            sim::SteinRepulsiveLangevin,
            n_steps::Integer,
            al::ActiveLearnRoutine;
            n_threads::Integer=Threads.nthreads(),
            run_loggers=true,
            rng=Random.GLOBAL_RNG,
)
    sys, neighbors = initialize_sim!(sys; n_threads=n_threads, run_loggers=run_loggers)
    compute_error_metrics!(al)


    for step_n in 1:n_steps
        neighbors, ksd = simulation_step!(sys,
                            sim,
                            step_n,
                            n_threads=n_threads,
                            neighbors=neighbors,
                            run_loggers=run_loggers,
                            rng=rng,
        )

        sys_new = remove_loggers(sys)
        sim.sys_fix = reduce(vcat, [sim.sys_fix[2:end], sys_new])

        # online active learning
        if trigger_activated(al.trigger; step_n=step_n, ksd=ksd)
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end

# ensemble (multi-trajectory) MD
function active_learn!(ens::Vector{<:System},
            sim::StochasticSVGD,
            n_steps::Integer,
            al::ActiveLearnRoutine;
            n_threads::Integer=Threads.nthreads(),
            run_loggers=true,
            rng=Random.GLOBAL_RNG,
)

    N = length(ens)
    T = typeof(find_neighbors(ens[1], n_threads=n_threads))
    nb_ens = Vector{T}(undef, N)
    bwd = zeros(n_steps)
    compute_error_metrics!(al)

    # initialize
    for (sys, nb) in zip(ens, nb_ens)
        sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
        !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
        nb = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
        run_loggers!(sys, nb, 0, run_loggers; n_threads=n_threads)
    end

    for step_n in 1:n_steps
        nb_ens, ksd, bwd[step_n] = simulation_step!(ens,
                    nb_ens,
                    sim,
                    step_n,
                    n_threads=n_threads,
                    run_loggers=run_loggers,
                    rng=rng,
        )

        # online active learning
        if trigger_activated(al.trigger; step_n=step_n, ens_old=al.sys_train, ens_new=ens, ksd=ksd)
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, ens, al.sys_train)
            al.train_func(ens, al.sys_train, al.ref) # retrain potential
            al.mlip = ens[1].general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [ens[1].general_inters[1].params])
            compute_error_metrics!(al)
        end
    end

    return al, bwd
end


function active_learn!(sys::System,
    sim::OverdampedLangevin,
    n_steps::Integer,
    al::ActiveLearnRoutine;
    n_threads::Integer=Threads.nthreads(),
    run_loggers=true,
    rng=Random.GLOBAL_RNG,
)
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    compute_error_metrics!(al)

    for step_n in 1:n_steps
        neighbors = simulation_step!(sys,
                        sim,
                        step_n,
                        n_threads=n_threads,
                        neighbors=neighbors,
                        run_loggers=run_loggers,
                        rng=rng,
        )

        # online active learning
        if trigger_activated(al.trigger; ens_old=al.sys_train, sys_new=sys, step_n=step_n)
            println("train on step $step_n")
            al.sys_train = al.update_func(sim, sys, al.sys_train)
            al.train_func(sys, al.sys_train, al.ref) # retrain potential
            al.mlip = sys.general_inters[1]
            append!(al.train_steps, step_n)
            append!(al.param_hist, [sys.general_inters[1].params])
            compute_error_metrics!(al)
        end
    end
    return al
end


function active_learn!(sys::System,
                       sim::OverdampedLangevin,
                       n_steps::Integer,
                       al::ALRoutine;
                       n_threads::Integer=Threads.nthreads(),
                       run_loggers=true,
                       rng=Random.GLOBAL_RNG)
    #println("simulation initialization:")
    neighbors =  initialize_sim!(sys,sim; n_threads=n_threads, run_loggers=run_loggers)
    #println("triggers initialization")
    sys = initialize_triggers(al.triggers,sys) # new System returned with updated loggers and data field
    #println("aldata initialization")
    aldata = initialize_al_record(al.aldata_spec,sys,al)

    for step_n in 1:n_steps
        al.cache[:step_n] = step_n
        #println("simulation step")
        neighbors = simulation_step!(sys,
                                     sim,
                                     step_n;
                                     n_threads=n_threads,
                                     neighbors=neighbors,
                                     run_loggers=run_loggers,
                                     rng=rng)

        # ideally will also return trigger_state to know which trigger is activated, etc.
        #println("trigger activation")
        trigg_res = trigger_activated!(al.triggers,sys, al)
        if trigg_res
            println("trigger activated on step $(step_n)")
            #al.cache[:trigger_state] = trigg_state

            # update trainset
            al.trainset, al.cache[:trainset_changes] = update_trainset!(al.ss,sys,al) # can modify caches
            # train new mlip
            @time al.mlip =  retrain!(al.lp,sys,al) # can modify caches
            println("training mlip complete")
            sys.general_inters = (al.mlip,)

            if !isnothing(al.trigger_updates)
                @time updated_triggers = update_triggers!(al.triggers, al.trigger_updates, sys, al) #trigger updates modified in place
                println("training committee complete")
                al.triggers = updated_triggers
            end

            if !isnothing(al.sim_update)
                update_simulator!(al.sim_update,sim,sys,al) # why not follow the same pattern of returning object then setting object on next line
            end

            record_al_record!(al.aldata_spec,aldata,sys,al)

            #can maybe wrap this in a function?
            al.cache[:trainset_changes] = nothing

            #reset_al_cache_after_train(al) # maybe should have something like this?
        end
        perstep_reset!(al.triggers,sys)
        al.cache[:step_n] = nothing
    end
    aldata, sys
end