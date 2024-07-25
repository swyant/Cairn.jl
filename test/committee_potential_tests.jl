using Cairn
using Molly #also reexports Unitful
using AtomsCalculators
import StaticArrays: SVector
import SpecialPolynomials: Jacobi
import AtomsBase: FlexibleSystem
using Test 

pce_template = PolynomialChaos(3, 2, Jacobi{0.5,0.5})

# randomly generated parameters
all_params = [[48.7321 95.9563 11.8879 3.56038 94.8206 51.6035 69.2488 37.2358 81.8482 40.592];
              [81.6018 30.8123 56.9185 98.2884 17.8753 2.37024 65.9343 12.8787 74.2257 12.8148];
              [6.65932 62.8262 0.61548 91.5065 76.1619 27.3688 91.6777 66.1589 34.1041 8.07007];
              [55.9341 15.572  19.6692 95.2555 5.81737 20.1318 68.1779 88.3704 6.86962 30.4034];
              [88.3364 62.2294 26.9018 39.6262 46.7406 99.4504 16.5021 38.1385 6.07858 66.3549]]

all_pces = [let 
                coeffs = all_params[i,:]
                pce = deepcopy(pce_template)
                pce.params = coeffs 
                pce 
            end
            for i in 1:5]

##Checking different constructors
cmte_pot1 = CommitteePotential(all_pces,3,u"kJ * mol^-1",u"nm")
@test typeof(cmte_pot1) <: CommitteePotential{<:PolynomialChaos}
cmte_pot2 = CommitteePotential(all_pces,3,u"eV",u"Å") # units don't match member PCE, but this is not enforced
@test typeof(cmte_pot2) <: CommitteePotential{<:PolynomialChaos}
cmte_pot3 = CommitteePotential(all_pces,3,NoUnits,NoUnits) # units don't match member PCE, but this is not enforced
@test typeof(cmte_pot3) <: CommitteePotential{<:PolynomialChaos}
cmte_pot4 = CommitteePotential(all_pces;energy_units=u"kJ * mol^-1",length_units= u"nm")
@test typeof(cmte_pot4) <: CommitteePotential{<:PolynomialChaos}
@test cmte_pot4.leader == 1
cmte_pot5 = CommitteePotential(all_pces) #adopts default units, which don't match member PCE units, but this is not enforced 
@test typeof(cmte_pot5) <: CommitteePotential{<:PolynomialChaos}
@test cmte_pot5.leader == 1
@test cmte_pot5.energy_units == u"eV"
@test cmte_pot5.length_units == u"Å"

#TODO: redo above and below with a different type of member potential, prefereably ACE

#Checking bad inputs
@test_throws "leader index is out of bounds" CommitteePotential(all_pces,7)
@test_throws "leader index is out of bounds" CommitteePotential(all_pces,-1)
@test_throws MethodError CommitteePotential(all_pces;energy_units=u"nm") #wrong unit dimensions
@test_throws MethodError CommitteePotential(all_pces;length_units=u"K") #wrong unit dimensions

# construct simple reference Molly system  
ref = MullerBrownRot() # not actually invoked, just a "neutral" potential for these AtomsCalculators tests below
xref = SVector{2}([0.096079218176701, -0.9623723102484034]) .* u"nm"
sys_ref = System(ref,xref)

#TODO use AtomsCalculators test suite

# checking against reference values that cmte_pot computes correct "leader" forces and energies
echeck1 = AtomsCalculators.potential_energy(sys_ref,cmte_pot1) #leader is 3
@test typeof(echeck1) <: typeof(1.0u"kJ * mol^-1")
@test echeck1 ≈ 4.6978599790506435u"kJ * mol^-1"

fcheck1 = AtomsCalculators.forces(sys_ref,cmte_pot1) # leader is 3
@test length(fcheck1) == 1 # only one atom, should test other conditions
@test typeof(fcheck1) <: Vector{SVector{2,typeof(1.0u"kJ * mol^-1 * nm^-1")}}
@test fcheck1[1] ≈ [-57.734110333280256, -12.508601878931005]u"kJ * mol^-1 * nm^-1"

# Molly equivalence test, i.e. running with cmte_pot w/ leader 3 should produce same traj as directly running with pce3
pce3 = deepcopy(pce_template)
pce3.params = all_params[3,:]
init_vels = random_velocities(sys_ref,300.0u"K")
sys_pce3 = Molly.System(sys_ref; 
                        general_inters =(pce3,),
                        velocities=init_vels,
                        force_units=u"kJ * mol^-1 * nm^-1", #want to be explicit with units just in case
                        energy_units=u"kJ * mol^-1")

sim_nh = NoseHoover(dt=0.002u"ps",          #deterministic simulator
                    temperature=300.0u"K", 
                    remove_CM_motion=1)

simulate!(sys_pce3,sim_nh,100)

sys_cmtepot1 = Molly.System(sys_ref;
                            general_inters =(cmte_pot1,),
                            velocities=init_vels,
                            force_units=u"kJ * mol^-1 * nm^-1", #want to be explicit with units just in case
                            energy_units=u"kJ * mol^-1")

simulate!(sys_cmtepot1,sim_nh,100)

@test position(sys_pce3)[1] ≈ position(sys_cmtepot1)[1]
@test forces(sys_pce3)[1] ≈ forces(sys_cmtepot1)[1]
                        