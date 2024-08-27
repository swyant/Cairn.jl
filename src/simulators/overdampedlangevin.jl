
function simulation_step!(sys::System,
    sim::OverdampedLangevin,
    step_n::Integer;
    n_threads::Integer=Threads.nthreads(),
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads),
    run_loggers=true,
    rng=Random.GLOBAL_RNG)

    forces_t = forces(sys, neighbors; n_threads=n_threads)
    accels_t = forces_t ./ masses(sys)

    noise = random_velocities(sys, sim.temperature; rng=rng)
    sys.coords += (accels_t ./ sim.friction) .* sim.dt .+ sqrt((2 / sim.friction) * sim.dt) .* noise
    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))

    if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
        remove_CM_motion!(sys)
    end

    neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                       n_threads=n_threads)

    run_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads,
                current_forces=forces_t)

    return neighbors
end

# initialize simulation
function initialize_sim!(sys::System,
                         sim::OverdampedLangevin;
                         n_threads::Integer=Threads.nthreads(),
                         run_loggers=true,
                        )

    if length(sys.constraints) > 0
        @warn "OverdampedLangevin is not currently compatible with constraints, " *
        "constraints will be ignored"
    end

    sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    nb = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    run_loggers!(sys, nb, 0, run_loggers; n_threads=n_threads)

    nb
end

# function simulate!(sys::System,
#     sim::OverdampedLangevin,
#     n_steps::Integer;
#     n_threads::Integer=Threads.nthreads(),
#     run_loggers=true,
#     rng=Random.GLOBAL_RNG)

#     sys.coords = wrap_coords.(sys.coords, (sys.boundary,))
#     !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
#     neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
#     run_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)

#     for step_n in 1:n_steps
#         neighbors = simulation_step!(sys,
#                             sim,
#                             step_n,
#                             n_threads=n_threads,
#                             neighbors=neighbors,
#                             run_loggers=run_loggers,
#                             rng=rng,
#         )
#     end

# end