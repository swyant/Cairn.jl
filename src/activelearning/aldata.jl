export  DefaultALDataSpec,
        IPErrors,
        Simple2DPotErrors,
        initialize_al_record,
        record_al_record!


###### InteratomicPotential Error Types

abstract type IPErrors end

# train force and energy rmse, fisher divergence parameterized by integrator
struct Simple2DPotErrors <: IPErrors
    eval_int # if nothing, won't compute fisher divergence (e.g., for committee potential)
    compute_fisher::Bool
end

function initialize_error_metrics!(error_metric_type::Simple2DPotErrors, ddict::Dict)
    ddict["error_hist"] = Dict("rmse_e" => [],
                               "rmse_f" => [])
    if error_metric_type.compute_fisher
        ddict["error_hist"]["fd"] = []
    end
end

# like this is nearly identical to what she did but it's more verbose, so you have to explain why do it like this
function record_errors!(error_metric_type::Simple2DPotErrors, aldata::Dict, sys::Molly.System, al)
    r_e, r_f = compute_rmse(al.ref, al.mlip, error_metric_type.eval_int) # invokes the grid in the integrator
    push!(aldata["error_hist"]["rmse_e"], r_e)
    push!(aldata["error_hist"]["rmse_f"], r_f)
    if error_metric_type.compute_fisher
        fd = compute_fisher_div(al.ref, al.mlip, error_metric_type.eval_int)
        push!(aldata["error_hist"]["fd"], fd)
    end
end

###### DefaultAlDataSpec
struct DefaultALDataSpec
    error_metrics::IPErrors # Needs to be a stuct because of the initialization
    record_trigger_step::Bool
    #record_trigger_res::Bool
    record_parameters::Bool
    record_new_configs::Bool
    record_trigger_logs::Bool # could in theory only record a few triggers, but whatever
end

function DefaultALDataSpec(error_metrics::IPErrors;
          record_trigger_step::Bool=true,
          record_parameters::Bool=true,
          record_new_configs::Bool=false,
          record_trigger_logs::Bool=false)

    new_aldata_spec = DefaultALDataSpec(error_metrics,
                                        record_trigger_step,
                                        record_parameters,
                                        record_new_configs,
                                        record_trigger_logs)
    new_aldata_spec
end


function initialize_al_record(al_spec::DefaultALDataSpec, sys, al)
    aldata = Dict()
    initialize_error_metrics!(al_spec.error_metrics,aldata)
    if al_spec.record_trigger_step
        aldata["trigger_steps"] = []
    end

    if al_spec.record_parameters
        #this way the starting state is stored, but the "mlip_params" field will be the same length as the other fields
        aldata["initial_mlip_params"] = get_params(al.mlip)
        aldata["mlip_params"] = []
    end

    if al_spec.record_new_configs
        aldata["new_configs"] = []
    end

    if al_spec.record_trigger_logs
        aldata["activated_trigger_logs"] = Dict()
        for trigger in al.triggers
            logger_ids = get_logger_ids(trigger) # can be more than one key
            filtered_logger_ids = filter(x->!isnothing(x), logger_ids)

            for logger_id in filtered_logger_ids
                aldata["activated_trigger_logs"][logger_id] = []
            end
        end
    end

    aldata
end


function record_al_record!(al_spec::DefaultALDataSpec, aldata::Dict, sys::Molly.System, al)
    record_errors!(al_spec.error_metrics, aldata, sys, al)

    if al_spec.record_trigger_step
        push!(aldata["trigger_steps"], al.cache[:step_n])
    end

    if al_spec.record_parameters
        #TODO need to formalize an interface to get parameters from mlips vs committee potentials
        params = get_params(al.mlip)
        push!(aldata["mlip_params"],params)
    end

    if al_spec.record_new_configs
        push!(aldata["new_configs"], al.cache[:trainset_changes])
    end

    if al_spec.record_trigger_logs
        for logger_key in keys(aldata["activated_trigger_logs"])
            observed_val = sys.loggers[logger_key].observable
            push!(aldata["activated_trigger_logs"][logger_key], observed_val)
        end
    end
end


