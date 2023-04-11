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
zmats_is0 = coord_maps.get_internal_from_trajectory(traj_is0, isomer=0, temperature=300)#, max_samples=10)
zmats_is0 = zmats_is0.detach()
zmats_is0[:, 11][zmats_is0[:, 11]>0] = zmats_is0[:, 11][zmats_is0[:, 11]>0].apply_(add_phase)

zmats_is1 = coord_maps.get_internal_from_trajectory(traj_is1, isomer=1, temperature=300)#, max_samples=10)

save_internal_coordinates_to_csv(zmats_is0, get_construction_table(), filename='int_coords/is0_md_zmat.csv')
save_internal_coordinates_to_csv(zmats_is1, get_construction_table(), filename='int_coords/is1_md_zmat.csv')