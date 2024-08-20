#=
TODO
- [ ] export AbstractLearningProblem in PL.jl
- [ ] Given tight coupling between cmte indices and the trainset, may want to include trainset as a field.
      Honestly, for safety reasons, may make sense to make SubsampleAppendCmteRetrain immutable?
=#
using Random: randperm

export InefficientLearningProblem,
       AbstractCmteLearningProblem,
       SubsampleAppendCmteRetrain,
       learn #different from learn!()


# Recomputing all descriptors, energies, forces for entire trainset
struct InefficientLearningProblem <: PotentialLearning.AbstractLearningProblem
    weights::Vector{Float64}
    intcpt::Bool
    ref
end

function InefficientLearningProblem(weights=[1000.0,1.0],intcpt=false; ref=nothing)
    return InefficientLearningProblem(weights,intcpt,ref)
end

function learn(ilp::InefficientLearningProblem, mlip, trainset; ref=ilp.ref)
    lp = learn!(trainset, ref, mlip, ilp.weights, ilp.intcpt; e_flag=true, f_flag=true)
    new_mlip = deepcopy(mlip) #How to generalize? Should mlip be modified in place
    new_mlip.params = lp.β

    new_mlip
end

abstract type AbstractCmteLearningProblem <: PotentialLearning.AbstractLearningProblem end

mutable struct SubsampleAppendCmteRetrain  <: AbstractCmteLearningProblem
    lp::PotentialLearning.AbstractLearningProblem #need to enforce concrete
    cmte_indices::Vector{Vector{Integer}}
    #trainset (this could be a field, somewhat justified given tight coupling between indices and corresponding dataset)
end

# nothing is modified in place here
function learn(clp::SubsampleAppendCmteRetrain,
               cmte_pot_template::CommitteePotential{P},
               new_trainset) where {P}
    new_members = Vector{P}()
    for (old_mlip, train_indices) in zip(cmte_pot_template.members, clp.cmte_indices)
        new_mlip = learn(clp.lp, old_mlip, new_trainset[train_indices])
        push!(new_members,new_mlip)
    end

    new_cmte_pot = CommitteePotential(new_members,
                                      cmte_pot_template.leader,
                                      cmte_pot_template.energy_units,
                                      cmte_pot_template.length_units)
    new_cmte_pot
end

# a bespoke function to instantiate cmte_mlip by first generating subsample indices, then performing learn!() on each member
# updates clp.cmte_indices in-place, returns cmte_pot
function learn!(clp::SubsampleAppendCmteRetrain,
                cmte_pot_template::CommitteePotential,
                trainset::Vector{<:AbstractSystem};
                frac=0.7,
                train_subset_idxs=nothing)

    #obtain subset indices for each cmte member, update clp
    new_indices = Vector{Int64}[]
    num_fits = length(cmte_pot_template.members)
    for fit_idx in 1:num_fits
        if isnothing(train_subset_idxs)
            train_idxs = obtain_train_idxs(frac, length(trainset))
        else
            train_idxs = obtain_train_idxs(frac, train_subset_idxs)
        end
        push!(new_indices,train_idxs)
    end
    clp.cmte_indices = new_indices

    #with updated committee indices, fit new committee potential
    cmte_pot = learn(clp,cmte_pot_template,trainset)
    cmte_pot
end


# CLP is updated in place (i.e, cmte_indices updated), learn!() new cmte_pot and return that
function learn!(clp::SubsampleAppendCmteRetrain,
                  cmte_pot::CommitteePotential,
                  num_new_configs::Integer,
                  new_trainset::Vector{<:AbstractSystem})

    new_trainset_size = length(new_trainset)
    append_indices = (new_trainset_size-num_new_configs+1):new_trainset_size

    # append new indices (new configurations) to each cmte index set
    updated_indices = []
    for old_indices in clp.cmte_indices
        new_indices = reduce(vcat, [old_indices, [x for x in append_indices]])
        push!(updated_indices,new_indices)
    end

    clp.cmte_indices = updated_indices
    new_cmte_pot = learn(clp,cmte_pot,new_trainset)

    new_cmte_pot
end


# generate random subset indices, given size of reference array
function obtain_train_idxs(frac::Float64, set_size::Int64)
    @assert frac <= 1.0
    num_select = Int(floor(frac*set_size))

    perm_idxs = randperm(set_size)
    rand_set_idxs = perm_idxs[begin:1:num_select]

    rand_set_idxs
end

#Given existing subset of indices, subsample further, but indices will continue to corresponding (superset) reference array
function obtain_train_idxs(frac::Float64, train_subset_idxs::Vector{Int64})
    @assert frac <= 1.0

    set_size = length(train_subset_idxs)
    num_select = Int(floor(frac*set_size))

    perm_idxs = randperm(set_size)
    rand_set_idxs = perm_idxs[begin:1:num_select]

    train_subset_idxs[rand_set_idxs]
end
