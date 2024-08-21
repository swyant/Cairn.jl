export GreedySelector

struct GreedySelector <: SubsetSelector
end

function update_trainset!(ss::GreedySelector,
                          sys::Molly.System,
                          al)
    new_sys = strip_system(sys)
    new_trainset = reduce(vcat, [al.trainset, new_sys])

    new_trainset, [new_sys,]
end

function strip_system(sys::Molly.System)
    stripped_sys = System(
                    atoms    = sys.atoms,
                    coords   = sys.coords,
                    boundary = sys.boundary)
    stripped_sys
  end