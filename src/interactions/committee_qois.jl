export AbstractCommitteeQoI,
       CmteEnergy,
       CmteFlatForces,
       CmteForces,
       compute #probably too generic of a name to export

abstract type AbstractCommitteeQoI end

### Committee Energy QoIs ###

struct CmteEnergy <: AbstractCommitteeQoI
    cmte_reduce::Union{Nothing,Function}
    strip_units::Bool

    function CmteEnergy(reduce_function::Union{Nothing,Function}, strip_units::Bool)
        if !isnothing(reduce_function)
            if !_check_reduction_fn(reduce_function)
                error("Reduction function invalid. Must reduce to single Float, Integer, or Boolean")

            end
        end
        return new(reduce_function,strip_units)
    end

end

function CmteEnergy(reduce_fn::Union{Nothing, Function}=nothing;
                    strip_units::Bool=false)
    cmte_e_qoi = CmteEnergy(reduce_fn,strip_units)
    cmte_e_qoi
end

function compute(qoi::CmteEnergy,sys::AbstractSystem,cmte_pot::CommitteePotential)

    all_energies = compute_all_energies(sys,cmte_pot)

    if qoi.strip_units
        if eltype(all_energies) <: Unitful.Quantity # may be unnecessary once AC interface firmly enforced?
            all_energies = ustrip.(all_energies)
        end
    end

    if !isnothing(qoi.cmte_reduce)
        reduced_energies = qoi.cmte_reduce(all_energies)
        return reduced_energies
    else
        return all_energies
    end
end





### Committee Flattened Forces QoIs ###

struct CmteFlatForces <: AbstractCommitteeQoI
    cmte_reduce::Union{Nothing,Function}
    coord_and_atom_reduce::Union{Nothing,Function}
    reduce_order::Vector{Int64}
    strip_units::Bool

    function CmteFlatForces(cmte_reduce_function::Union{Nothing,Function},
                            coord_and_atom_reduce_function::Union{Nothing,Function},
                            reduce_order::Vector{Int64},
                            strip_units::Bool)
        cmte_reduce_valid = true
        num_reduce_fns = 0
        if !isnothing(cmte_reduce_function)
            num_reduce_fns += 1
            if !_check_reduction_fn(cmte_reduce_function)
                cmte_reduce_valid = false
            end
        end

        coord_and_atom_reduce_valid = true
        if !isnothing(coord_and_atom_reduce_function)
            num_reduce_fns += 1
            if !_check_reduction_fn(coord_and_atom_reduce_function)
                coord_and_atom_reduce_valid = false
            end
        end

        if !cmte_reduce_valid && coord_and_atom_reduce_valid
            error("Cmte reduction function invalid. Must reduce to single Float, Integer, or Boolean")
        elseif cmte_reduce_valid && !coord_and_atom_reduce_valid
            error("Coord_and_atom reduction function invalid. Must reduce to single Float, Integer, or Boolean")
        elseif !cmte_reduce_valid && !coord_and_atom_reduce_valid
            error("Cmte reduction and coord_and_atom reduction functions invalid. Must reduce to single Float, Integer, or Boolean")
        end

        if length(reduce_order) > num_reduce_fns || maximum(reduce_order) > 2 || minimum(reduce_order) < 1
            error("Invalid reduce_order. Please use NamedTuple-based constructor")
        end

        return new(cmte_reduce_function,coord_and_atom_reduce_function,reduce_order,strip_units)
    end

end

# There is some argument to be made that this NamedTuple-based constructor should be the inner constructor
function CmteFlatForces(nt::NamedTuple{<:Any, <:Tuple{Vararg{Function}}};
                        strip_units::Bool=false)
    fn_dict = Dict(:cmte =>
                    Dict{String,Union{Nothing,Function,Int64}}(
                        "fn"  => nothing,
                        "idx" => 1),
                    :coord_and_atom =>
                    Dict{String,Union{Nothing,Function,Int64}}(
                        "fn"  => nothing,
                        "idx" => 2)
                    )

    if !all(in.(keys(nt),[keys(fn_dict)]))
      error("""Only allowed keys are "cmte", "coord_and_atom" """)
    elseif length(nt) > 2
      error("There can be a maximum of 2 elements in the passed NamedTuple")
    end

    reduce_order = Int64[]
    for (k,fn) in pairs(nt)
      push!(reduce_order,fn_dict[k]["idx"])
      fn_dict[k]["fn"] = fn
    end

    cmte_flat_force_qoi = CmteFlatForces(fn_dict[:cmte]["fn"],
                                         fn_dict[:coord_and_atom]["fn"],
                                         reduce_order,
                                         strip_units)
    cmte_flat_force_qoi
end

function CmteFlatForces()
    flat_force_qoi = CmteFlatForces(nothing,nothing,Int64[],false)
    flat_force_qoi
end

function compute(qoi::CmteFlatForces,sys::AbstractSystem,cmte_pot::CommitteePotential)
    reduce_dict = Dict{Int64, Union{Nothing,Function}}(
                  1 => qoi.cmte_reduce,
                  2 => qoi.coord_and_atom_reduce)

    raw_all_forces = compute_all_forces(sys,cmte_pot)

    all_forces = stack(map(elem->stack(elem,dims=1),raw_all_forces), dims=1)  # num_cmte x num_atoms x 3
    all_flat_forces = reshape(permutedims(all_forces,(1,3,2)), size(all_forces,1), :) #num_cmte x num_atoms*3 1x,1y,1z,2x,2y,2z,etc.

    if qoi.strip_units
        if eltype(all_flat_forces) <: Unitful.Quantity # may be unnecessary once AC interface firmly enforced?
            all_flat_forces = ustrip.(all_flat_forces)
        end
    end

    if isnothing(qoi.cmte_reduce) && isnothing(qoi.coord_and_atom_reduce)
      return all_flat_forces
    else
      inter = all_flat_forces
      for d in qoi.reduce_order
        inter = mapslices(reduce_dict[d],inter,dims=d)
      end

      if length(qoi.reduce_order) == 2
        # assert statement fails with units
        #@assert size(inter) == (1,1) && typeof(inter[1]) <: Union{<:Real, <:Integer, Bool}
        final_qoi = inter[1]
      else
        for d in qoi.reduce_order
          @assert size(inter,d) == 1 # ensure singleton dimension
        end

        #arguably should check if Int,bool, float, but once I put that in the inner constructor it's fine
        final_qoi = dropdims(inter,dims=Tuple(qoi.reduce_order))
      end
      return final_qoi
    end
end






### Utilities

# only run @ initialization of committee QoI
function _check_reduction_fn(fn::Function)
  test_arr1 = [1.0,2.0,3.0,4.0,5.0] # usually vector being operated on will be floats

  local res = nothing # to deal with the scoping in the try statement, can now use res in else
  try
    res = fn(test_arr1)
  catch e
    if isa(e,Unitful.DimensionError)
        expected_units = unit(e.x)
        unitted_test_arr = test_arr1 * expected_units
        try
            res = fn(unitted_test_arr)
        catch
        else
            # ustrip should still work even if res is an array (and then check_res == false)
            # so don't need to proactively check that
            check_res = typeof(ustrip(res)) <: Union{<:Real, <:Integer, Bool}
            return check_res
        end
    end
  else
    check_res = typeof(res) <: Union{<:Real, <:Integer, Bool}
    return check_res
  end

  test_arr2 = [1,2,3,4,5] # if prior reduction step produced integers
  try
    res = fn(test_arr2)
  catch
  else
    check_res = typeof(res) <: Union{<:Real, <:Integer, Bool}
    return check_res
  end

  test_arr3 = [true,true,false,true,false] # if prior reduction step produced integers
  try
    res = fn(test_arr3)
  catch
  else
    check_res = typeof(res) <: Union{<:Real, <:Integer, Bool}
    return check_res
  end

  false
end