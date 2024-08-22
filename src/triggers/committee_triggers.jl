export  CmteTrigger,
        SharedCmteTrigger,
        trigger_activated!

###### CmteTrigger Implementation

struct CmteTrigger{Q <:AbstractCommitteeQoI, F<:Function, T} <: ActiveLearningTrigger
    cmte_qoi::Q
    compare::F
    thresh::T
    cmte_pot::Union{Nothing,CommitteePotential}
    logger_spec::Union{Nothing, Tuple{Symbol,Int64}}

    function CmteTrigger(cmte_qoi::Q,
                         compare::F,
                         thresh::T,
                         cmte_pot=nothing,
                         logger_spec=nothing
                         ) where {Q <:AbstractCommitteeQoI, F<:Function, T}
        if !(T <: Union{<:Real, <:Integer, Bool, Quantity})
            error("Threshold in CmteTrigger needs to be a float, integer, boolean, or Unitful.Quantity")
        end

        if !isnothing(logger_spec)
            if !(typeof(logger_spec) <: Tuple{Symbol,Int64})
                error("logger_spec needs to be Tuple{Symbol,Int64}, where the symbol is logger label, the int is the logger history frequency")
            end
        end

        return new{Q,F,T}(cmte_qoi,compare,thresh,cmte_pot,logger_spec)
    end

end

# The purpose of this extra constructor is to enable logger_spec to be a keyword argument,
# but it doesn't have to be (using the above inner constructor)
function CmteTrigger(cmte_qoi,
                     compare,
                     thresh;
                     cmte_pot=nothing,
                     logger_spec=nothing)
    cmte_trigger = CmteTrigger(cmte_qoi,compare,thresh,cmte_pot,logger_spec)
    cmte_trigger
end

# need to use this trick https://stackoverflow.com/questions/40160120/generic-constructors-for-subtypes-of-an-abstract-type
function (::Type{CmteTrigger{Q,F,T}})(trigger::CmteTrigger{Q,F,T};
                                      cmte_qoi::Q=trigger.cmte_qoi,
                                      compare::F=trigger.compare,
                                      thresh::T=trigger.thresh,
                                      cmte_pot::Union{Nothing,CommitteePotential}=trigger.cmte_pot,
                                      logger_spec::Union{Nothing,Tuple{Symbol,Int64}}=trigger.logger_spec) where {Q<:AbstractCommitteeQoI, F<:Function, T}
    #cmte_trigger = CmteTrigger{Q,F,T}(cmte_qoi,compare,thresh,cmte_pot,logger_spec) # compiler can't figure this out?
    cmte_trigger = CmteTrigger(cmte_qoi,compare,thresh,cmte_pot,logger_spec)
    cmte_trigger
end

# returns updated loggers, which is used to create an updated System copy in `initialize_triggers`
function append_loggers(trigger::CmteTrigger{Q,F,T}, loggers::NamedTuple) where {Q,F,T}

    if !isnothing(trigger.logger_spec)
        prior_keys = keys(loggers)
        prior_vals = values(loggers)

        log_symb = trigger.logger_spec[1]
        if log_symb in prior_keys
            error("Symbol provided in trigger.logger_spec is already used in the System.loggers")
        end
        log_freq = trigger.logger_spec[2]

        new_logger = SimpleTriggerLogger(log_freq,T)

        updated_keys = (prior_keys...,log_symb)
        updated_vals = (prior_vals...,new_logger)

        return NamedTuple{updated_keys}(updated_vals)
    else
        return loggers
    end
end

function trigger_activated!(trigger::CmteTrigger{Q,F,T},
                            sys::Molly.System,
                            al;
                            shared_cmte_pot::Union{Nothing,CommitteePotential}=nothing,
                            cache_field=nothing) where {Q,F,T}
    step_n = al.cache[:step_n]
    #Select commitee potential : shared_cmte_pot > trigger.cmte_pot > sys.general_inters[1]
    if !isnothing(shared_cmte_pot)
        cmte_pot = shared_cmte_pot # prefer over CmteTrigger.cmte_pot
    elseif !isnothing(trigger.cmte_pot)
        cmte_pot = trigger.cmte_pot
    elseif typeof(sys.general_inters[1]) <: CommitteePotential
        cmte_pot = sys.general_inters[1]
    else
        error("No committee potential available for trigger activation")
    end

    cmte_qoi = compute(trigger.cmte_qoi,sys,cmte_pot;cache_field=cache_field)
    if !(typeof(cmte_qoi) <: typeof(trigger.thresh))
        error("return type of compute(cmte_qoi,...) is not the same type as the threshold")
    end

    if !isnothing(trigger.logger_spec)
        log_property!(sys.loggers[trigger.logger_spec[1]], cmte_qoi,step_n)
    end

    res = trigger.compare(cmte_qoi,trigger.thresh)
    res
end

function reset_logger!(trigger::CmteTrigger, sys::Molly.System)
    if !isnothing(trigger.logger_spec)
        reset_observable!(sys.loggers[trigger.logger_spec[1]])
    end
end

function get_logger_ids(trigger::CmteTrigger; from_shared=false)
    if !isnothing(trigger.logger_spec)
        if from_shared
            return trigger.logger_spec[1]
        else
            return (trigger.logger_spec[1],)
        end
    else
        return nothing
    end
end

###### SharedCmteTrigger Implementation

struct SharedCmteTrigger <: ActiveLearningTrigger
    subtriggers::Tuple{Vararg{CmteTrigger}}
    cmte_pot::Union{Nothing,CommitteePotential}
    energy_cache_field::Union{Nothing,Symbol}
    force_cache_field::Union{Nothing,Symbol}

    function SharedCmteTrigger(subtriggers::Tuple{Vararg{CmteTrigger}},
                               cmte_pot::Union{Nothing,CommitteePotential}= nothing,
                               energy_cache_field::Union{Nothing,Symbol} = nothing,
                               force_cache_field::Union{Nothing,Symbol} = nothing)
        shared_cmte_trigger = new(subtriggers,cmte_pot,energy_cache_field, force_cache_field)
        shared_cmte_trigger
    end
end

# Same as inner constructor, but cache fields are keywords
# however when I explicitly typed the arguments, it said I was overwriting the inner constructor which surprised me
# TODO: need to check this
function SharedCmteTrigger(subtriggers,
                           cmte_pot;
                           energy_cache_field = nothing,
                           force_cache_field = nothing)
    shared_cmte_trigger = SharedCmteTrigger(subtriggers,cmte_pot,energy_cache_field, force_cache_field)
    shared_cmte_trigger
end

# I'm probably over-typing the arguments here. All the typing is taken care of by the inner constructor.
function SharedCmteTrigger(trigger::SharedCmteTrigger;
                           cmte_pot::Union{Nothing,CommitteePotential}=trigger.cmte_pot,
                           subtriggers::Tuple{Vararg{CmteTrigger}}=trigger.subtriggers,
                           energy_cache_field::Union{Nothing,Symbol} = trigger.energy_cache_field,
                           force_cache_field::Union{Nothing,Symbol} = trigger.force_cache_field)
    shared_cmte_trigger = SharedCmteTrigger(subtriggers,cmte_pot, energy_cache_field, force_cache_field)
    shared_cmte_trigger
end

function append_loggers(shared_trigger::SharedCmteTrigger, loggers::NamedTuple)
    for subtrigger in shared_trigger.subtriggers
        loggers = append_loggers(subtrigger,loggers)
    end
    return loggers
end


function initialize_data(shared_trigger::SharedCmteTrigger, ddict::Dict)
    ecache_field = shared_trigger.energy_cache_field
    fcache_field = shared_trigger.force_cache_field

    if !isnothing(ecache_field) || !isnothing(fcache_field)
        if !(:_reset_every_step in keys(ddict))
            ddict[:_reset_every_step] = Symbol[]
        end

        if !isnothing(ecache_field)
            if ecache_field in keys(ddict)
                error("energy_cache_field symbol already used as System.data dictionary key")
          end

            ddict[ecache_field] = nothing
            push!(ddict[:_reset_every_step],ecache_field)
        end

        if !isnothing(fcache_field)
            if fcache_field in keys(ddict)
                error("force_cache_field symbol already used as System.data dictionary key")
            end

            ddict[fcache_field] = nothing
            push!(ddict[:_reset_every_step],fcache_field)
        end
        return ddict
    else
        return ddict
    end
end


function trigger_activated!(shared_trigger::SharedCmteTrigger, sys::Molly.System, al)

    # Assuming shared cmte pot is either referenced in SharedCmteTrigger (first priority), or is sys.general_inters[1]
    if !isnothing(shared_trigger.cmte_pot)
        shared_cmte_pot = shared_trigger.cmte_pot
    else
        if typeof(sys.general_inters[1]) <: CommitteePotential
            shared_cmte_pot = sys.general_inters[1] # this explicitly overrides subtrigger.cmte_pot
        else
            error("SharedCmteTrigger cannot be used if neither SharedCmteTrigger.cmte_pot is set nor sys.general_inters[1] is a committee potential")
        end
    end


    all_res = Bool[]
    for subtrigger in shared_trigger.subtriggers
      #How to pass appropriate cache field for custom Committee QoIs
        if typeof(subtrigger.cmte_qoi) <: Union{CmteForces, CmteFlatForces}
            res = trigger_activated!(subtrigger,sys, al; shared_cmte_pot=shared_cmte_pot, cache_field=shared_trigger.force_cache_field)
        elseif typeof(subtrigger.cmte_qoi) <: CmteEnergy
            res = trigger_activated!(subtrigger,sys, al; shared_cmte_pot=shared_cmte_pot, cache_field=shared_trigger.energy_cache_field)
        else
            res = trigger_activated!(subtrigger,sys, al; shared_cmte_pot=shared_cmte_pot.cmte_pot)
        end
        push!(all_res, res)
    end

    final_res = any(all_res)
    final_res
end

function reset_logger!(shared_trigger::SharedCmteTrigger, sys::Molly.System)
    for subtrigger in shared_trigger.subtriggers
        reset_logger!(subtrigger,sys)
    end
end

function get_logger_ids(shared_trigger::SharedCmteTrigger)
    trigger_ids = [get_logger_ids(subtrigger;from_shared=true)
                  for subtrigger in shared_trigger.subtriggers]
    Tuple(trigger_ids)
end


function update_trigger!(update::SubsampleAppendCmteRetrain,
                         cmte_trigger::Union{CmteTrigger,SharedCmteTrigger};
                         al,
                         kwargs...)
    old_cmte_pot = cmte_trigger.cmte_pot
    if isnothing(old_cmte_pot)
        @warn "Not actually updating CmteTrigger, assuming committee potential used for sys.general_inters and updated with retrain!()"
        return cmte_trigger
    else
        new_trainset = al.trainset
        num_new_configs = length(al.cache[:trainset_changes]) # hard assumption that this is just a list of new systems

        new_cmte_pot = learn!(update, old_cmte_pot, num_new_configs, new_trainset) # clp modified in place

        #is this kind of notation discouraged?
        new_cmte_trigger = typeof(cmte_trigger)(cmte_trigger; cmte_pot=new_cmte_pot)

        return new_cmte_trigger
    end
end