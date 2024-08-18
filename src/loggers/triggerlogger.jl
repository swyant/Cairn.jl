import Molly: GeneralObservableLogger, log_property!

export TriggerLogger, SimpleTriggerLogger

"""
    TriggerLogger(trigger::ActiveLearningTrigger, nsteps::Int, history::Vector{T})

A logger which holds a record of evaluations of the trigger function for active learning.

# Arguments
- `trigger::ActiveLearningTrigger`      : trigger function.
- `observable::T`                       : value of the trigger function of type `T`.
- `n_steps::Int`                        : time step interval at which the trigger function is evaluated.
- `history::Vector{T}`                  : record of the trigger function evaluation.
"""
mutable struct TriggerLogger{A, T}
    trigger::A
    observable::T
    n_steps::Int
    history::Vector{T}
end


function TriggerLogger(trigger::ActiveLearningTrigger, T::DataType, n_steps::Integer)
    return TriggerLogger{typeof(trigger), T}(trigger, T[], n_steps, T[])
end
TriggerLogger(trigger::ActiveLearningTrigger, n_steps::Integer) = TriggerLogger(trigger, Float64, n_steps)


Base.values(logger::TriggerLogger) = logger.history


function log_property!(logger::TriggerLogger, s::System, neighbors=nothing,
    step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)

    obs = logger.trigger.eval(s)
    logger.observable = obs

    if (step_n % logger.n_steps) == 0
        if typeof(logger.trigger) <: Union{Bool, TimeInterval}
            return
        else
            push!(logger.history, obs)
        end
    end
end


function Base.show(io::IO, fl::TriggerLogger)
    print(io, "TriggerLogger{", eltype(fl.trigger), ", ", eltype(eltype(values(fl))), "} with n_steps ",
            fl.n_steps, ", ", length(values(fl)), " frames recorded for ",
            length(values(fl)) > 0 ? length(first(values(fl))) : "?", " atoms")
end


mutable struct SimpleTriggerLogger{T}
    observable::Union{Nothing,T}
    n_steps::Int
    history::Vector{T}
end

# if n_steps=0, only observable is recorded for the length of that timestep
function SimpleTriggerLogger(n_steps::Integer=0, T::DataType=Float64)
    return SimpleTriggerLogger{T}(nothing, n_steps, T[])
end

Base.values(logger::SimpleTriggerLogger) = logger.history

#skip the standard log_property!() call
function log_property!(logger::SimpleTriggerLogger, s::System, neighbors=nothing,
                       step_n::Integer=0; n_threads::Integer=Threads.nthreads(), kwargs...)
end

function log_property!(logger::SimpleTriggerLogger{T}, obs::T, step_n::Integer=0) where {T}
  logger.observable = obs
  if logger.n_steps != 0 && (step_n % logger.n_steps) == 0
        push!(logger.history, obs)
  end
end

function reset_observable!(logger::SimpleTriggerLogger)
  logger.observable = nothing
end

function Base.show(io::IO, fl::SimpleTriggerLogger)
    print(io, "SimpleTriggerLogger{", eltype(fl.history),"} with n_steps ",
          fl.n_steps, ", ", length(values(fl)), " frames recorded")
end
