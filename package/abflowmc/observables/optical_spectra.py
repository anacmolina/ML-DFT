from gpaw import GPAW, Mixer
from gpaw.mpi import world
from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.lrtddft2.lr_communicators import LrCommunicators
from ase.parallel import paropen


def write_ground_state(atoms, path, calc=None):
    """
    Compute and write the ground state of the system
    Args:
        atoms: ASE atoms object
        path: path to save the output
        calc: calculator object
    """

    if calc is None:
        calc = GPAW(h=0.2,
                nbands=-30,
                xc='PBE',
                #poissonsolver={'remove_moment': 1 + 3 + 5},
                poissonsolver={'name': 'MomentCorrectionPoissonSolver',
                               'poissonsolver': 'fast',
                               'moment_corrections': 1 + 3 + 5},
                convergence={'bands': -8},
                txt=path+'/gs.out',
                mixer=Mixer(beta=0.1),
	        eigensolver = "cg")

    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(path+'/gs.gpw')

def unoccupied_states(path):
    """
    Compute unoccupied states of the system
    Args:
        path: path to load the output from the ground state
    """

    calc = GPAW(path+'/gs.gpw')
    
    calc = calc.fixed_density(nbands=-30,
                          convergence={'bands': "all"},
                          maxiter=1000,
                          txt=path+'/unocc.out',
                          symmetry="off"
                          )
    
    calc.write(path+'/unocc.gpw', mode='all')


def linear_response(path):
    """
    Compute the linear response of the system
    Args:
        path: path to load the output from the unoccupied states
    """

    max_energy_diff = 5.5 #eV
    w=0.08

    dd_size = 2 * 2 * 2
    eh_size = world.size // dd_size
    assert eh_size * dd_size == world.size
    lr_comms = LrCommunicators(world, dd_size, eh_size)

    calc = GPAW(path+'/unocc.gpw',
            communicator=lr_comms.dd_comm)
    
    lr = LrTDDFT2(path+'/lr2', calc,
              fxc='LDA',
              max_energy_diff=max_energy_diff,
              recalculate=None,  # Change this to force recalculation
              lr_communicators=lr_comms,
              txt=path+'/lr2_with_%05.2feV.out' % max_energy_diff)

    # This is the expensive part triggering the calculation!
    lr.calculate()

    spec = lr.get_spectrum(path+'/spectrum_with_%05.2feV.dat'
                       % max_energy_diff,
                       min_energy=0,
                       max_energy=10,
                       energy_step=0.01,
                       width=w)

    # Get and write transitions
    trans = lr.get_transitions(path+'/transitions_with_%05.2feV.dat'
                           % max_energy_diff,
                           min_energy=0.0,
                           max_energy=10.0)

def lr_get_transitions(path):
    """
    Get the transitions of the system
    Args:
        path: path to load the output from the linear response
    """

    max_energy_diff = 5.5  # eV

    calc = GPAW(path+'/unocc.gpw')
    lr = LrTDDFT2(path+'/lr2', calc,
                  fxc='LDA',
                  max_energy_diff=max_energy_diff,
                  txt='-')
    
    spec = lr.get_spectrum(#'spectrum_with_%05.2feV.dat'
                           #% max_energy_diff,
                           min_energy=0,
                           max_energy=10,
                           energy_step=0.01,
                           width=0.1)

    # Get and write transitions
    trans = lr.get_transitions(#'transitions_with_%05.2feV.dat'
                               #% max_energy_diff,
                               min_energy=0.0,
                               max_energy=10.0)

    # Get and write transition contributions
    indexlist = [i for i in range(0,20)]
    for index in indexlist:
        f2 = lr.get_transition_contributions(index_of_transition=index)
        with paropen(path+'/tc_%03d_with_%05.2feV.txt'
                 % (index, max_energy_diff), 'w') as f:
            f.write('Transition %d at %.2f eV\n' % (index, trans[0][index]))
            f.write(' %5s => %5s  contribution\n' % ('occ', 'unocc'))
            for (ip, val) in enumerate(f2):
                if (val > 1e-3):
                    f.write(' %5d => %5d  %8.4f%%\n' %
                        (lr.ks_singles.kss_list[ip].occ_ind,
                         lr.ks_singles.kss_list[ip].unocc_ind,
                         val / 2. * 100))

def compute_optical_spectra(atoms, path):
    """
    Compute the optical spectra of the system
    Args:
        atoms: ASE atoms object
        path: path to save the output
    """

    write_ground_state(atoms, path)
    unoccupied_states(path)
    linear_response(path)
    lr_get_transitions(path)


    




    
