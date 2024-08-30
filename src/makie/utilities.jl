import AtomsCalculators: potential_energy
export coord_grid_2d,
       potential_grid_2d,
       potential #TODO this is too valuable of a name to export for this utility


## grid on 2d domain
function coord_grid_2d(
    limits::Vector{<:Vector},
    step::Real;
    dist_units = u"nm"
)
    xcoord = Vector(limits[1][1]:step:limits[1][2]) .* dist_units
    ycoord = Vector(limits[2][1]:step:limits[2][2]) .* dist_units
    return [xcoord, ycoord]
end


## generic potential energy function with coord as argument
function potential(inter, coord::SVector{2})
    sys = let coords=[coord]; () -> [SVector{2}(coords)]; end # pseudo-struct
    return potential_energy(sys, inter)
end


## grid across potential energy surface below cutoff
function potential_grid_2d(
    inter,
    limits::Vector{<:Vector},
    step::Real;
    cutoff = nothing,
    dist_units = u"nm",
)
    rng1, rng2 = coord_grid_2d(limits, step; dist_units=dist_units)
    coords = SVector[]

    for i = 1:length(rng1)
        for j = 1:length(rng2)
            coord = SVector{2}([rng1[i],rng2[j]])
            Vij = ustrip(potential(inter, coord))
            if typeof(cutoff) <: Real && Vij <= cutoff
                append!(coords, [coord])
            elseif typeof(cutoff) <: Vector && cutoff[1] <= Vij <= cutoff[2]
                append!(coords, [coord])
            end
        end
    end

    return coords
end
