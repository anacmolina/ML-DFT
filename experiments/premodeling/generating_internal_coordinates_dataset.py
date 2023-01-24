from ase.io.trajectory import Trajectory

from flonacomldft.utils.io_utils import get_path
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    Coordinates_mapping,
    save_internal_coordinates_to_csv
)

traj_is0 = Trajectory(get_path() + 'md_trajectories/' + 'ag6_is0_lcao.traj') 
traj_is1 = Trajectory(get_path() + 'md_trajectories/' + 'ag6_is1_lcao.traj') 

coord_maps = Coordinates_mapping()
xs_is0 = coord_maps.get_internal_from_trajectory(traj_is0, temperature=300)
xs_is0 = xs_is0.detach()
xs_is0[:, 11][xs_is0[:, 11]>0] = xs_is0[:, 11][xs_is0[:, 11]>0].apply_(add_phase)

xs_is1 = coord_maps.get_internal_from_trajectory(traj_is1, temperature=300)

# TODO: Add isomer flag to the file

save_internal_coordinates_to_csv(xs_is0, get_construction_table(), filename='int_coords/is0_zmat.csv')
save_internal_coordinates_to_csv(xs_is1, get_construction_table(), filename='int_coords/is1_zmat.csv')