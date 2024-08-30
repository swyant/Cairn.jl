export
    ActiveLearningTrigger,
    trigger_activated,
    initialize_triggers,
    perstep_reset!,
    update_triggers!

"""
Abstract type for defining criteria triggering the active learning step during simulation.

"""
abstract type ActiveLearningTrigger end



"""
    trigger_activated(trigger::ActiveLearningTrigger, kwargs...)
    trigger_activated(trigger::Bool, kwargs...)

A function which returns a Bool of whether or not the trigger for active learning is activated.
"""
trigger_activated(sys, trigger::Bool; kwargs...) = trigger


## evaluate multiple triggers in order
function trigger_activated(
    sys::Union{System, Vector{<:System}},
    triggers::Tuple{<:ActiveLearningTrigger};
    sys_train=nothing,
    step_n::Integer=1,
)
    for trigger in triggers
        if trigger_activated(sys, trigger; sys_train=sys_train, step_n=step_n)
            return true
        end
    end
end

# Alternative simple multiple trigger evaluation
function trigger_activated!(triggers::Tuple{<:ActiveLearningTrigger}, sys::Molly.System, al)

    # this approach runs through all triggers, even if an earlier trigger already returns true
    # this ensures any later SimpleTriggerLoggers get logged
    all_res = Bool[]
    for trigger in triggers
        res = trigger_activated!(trigger,sys,al)
        push!(all_res,res)
    end

    final_res = any(all_res)
    final_res
end

function initialize_triggers(triggers::Tuple, sys::Molly.System)
    if typeof(sys.loggers) <: Tuple && length(sys.loggers) == 0  # if loggers is empty Tuple, convert to empty NamedTuple
        loggers = NamedTuple()
    elseif typeof(sys.loggers) <: NamedTuple
        loggers = sys.loggers
    else
        error("Can't handle a case where sys.loggers is Tuple with finite size, or anything other than NamedTuple")
    end

    if isnothing(sys.data)
        ddict = Dict{Any,Any}() #have to be flexible with types, user can do anything
    elseif typeof(sys.data) <: Dict
        ddict = Dict{Any,Any}(sys.data) # existing data dict may be too strictly typed
    else
        error("System.data needs to be either nothing or a dictionary")
    end

    for trigger in triggers
        loggers = append_loggers(trigger,loggers)
        ddict   = initialize_data(trigger,ddict)
    end

    return_sys = Molly.System(sys; loggers=loggers, data=ddict)

    return_sys
end

# default does nothing
function initialize_data(trigger,ddict)
    return ddict
end

# default does nothing
function append_loggers(trigger,loggers)
    return loggers
end

function perstep_reset!(triggers, sys::Molly.System)
  # need an API to get whether trigger has associated logger and what the logger name is
  for trigger in triggers
      reset_logger!(trigger,sys)
  end

  if (typeof(sys.data) <: Dict &&
      :_reset_every_step in keys(sys.data) &&
      length(sys.data[:_reset_every_step]) > 0)

      for dict_symb in sys.data[:_reset_every_step]
          sys.data[dict_symb] = nothing
      end
  end
end

# default does nothing here, but probably should be a standardized default way of
# registering and resetting trigger loggers based on a logger_spec field (or get_logger_id())
function reset_logger!(trigger, sys::Molly.System)
end

function update_triggers!(triggers, updates, sys::Molly.System, al)
    new_triggers = ActiveLearningTrigger[]
    for (up,trigg) in zip(updates,triggers)
        if !isnothing(up)
            new_trigg = update_trigger!(up, trigg; sys=sys,al=al)
            push!(new_triggers,new_trigg)
        else
            push!(new_triggers,trigg)
        end
    end

    return Tuple(new_triggers)
end

include("timeinterval.jl")
include("maxkernel.jl")
include("meanksd.jl")
include("maxvol.jl")
include("committee_triggers.jl")