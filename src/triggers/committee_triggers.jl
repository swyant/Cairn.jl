export CmteTrigger, trigger_activated!

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
                            al,
                            step_n::Integer=0;
                            shared_cmte_pot::Union{Nothing,CommitteePotential}=nothing,
                            cache_field=nothing) where {Q,F,T}

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

    cmte_qoi = compute(trigger.cmte_qoi,sys,cmte_pot)
    if !(typeof(cmte_qoi) <: typeof(trigger.thresh))
        error("return type of compute(cmte_qoi,...) is not the same type as the threshold")
    end

    if !isnothing(trigger.logger_spec)
        log_property!(sys.loggers[trigger.logger_spec[1]], cmte_qoi,step_n)
    end

    res = trigger.compare(cmte_qoi,trigger.thresh)
    res
end