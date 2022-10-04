import sys

from gpaw.mpi import world
from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.lrtddft2.lr_communicators import LrCommunicators

# Maximum energy difference for Kohn-Sham transitions
# included in the calculation
max_energy_diff = 5.5  # eV
w = 0.08

# Set parallelization
#dd_size = 2 * 1 * 1
#eh_size = world.size // dd_size
#assert eh_size * dd_size == world.size
#lr_comms = LrCommunicators(world, dd_size, eh_size)

i = int(sys.argv[1])

calc = GPAW('unocc_'+str(i)+'.gpw')#,
#communicator=lr_comms.dd_comm)
lr = LrTDDFT2('lr2', calc,
              fxc='PBE',
              max_energy_diff=max_energy_diff,
              recalculate=None,  # Change this to force recalculation
                  #              lr_communicators=lr_comms,
              txt='lr2_with_%0.1feV_%d.out' % (max_energy_diff, i))

# This is the expensive part triggering the calculation!
lr.calculate()

# Get and write spectrum
spec = lr.get_spectrum('spectrum_with_%0.1feV_%d.dat'
                       % (max_energy_diff, i),
                       min_energy=0,
                       max_energy=10,
                       energy_step=0.01,
                       width=w)

# Get and write transitions
trans = lr.get_transitions('transitions_with_%0.1feV_%d.dat'
                           %(max_energy_diff, i),
                           min_energy=0.0,
                           max_energy=10.0)