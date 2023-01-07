from ase.io.trajectory import Trajectory

from flonacomldft.utils.io_utils import get_path
from flonacomldft.internal_coordinates import (
    add_phase,
    get_construction_table,
    get_internal_coordinates_from_trajectory,
    save_internal_coordinates_to_csv
)

traj_is1 = Trajectory(get_path() + 'MD_trajectories/' + 'ag6_is1_lcao.traj') 
traj_is2 = Trajectory(get_path() + 'MD_trajectories/' + 'ag6_is2_lcao.traj') 

xs_is1 = get_internal_coordinates_from_trajectory(traj_is1, get_construction_table(), temperature=300)
xs_is1 = xs_is1.detach()
xs_is1[:, 11][xs_is1[:, 11]>0] = xs_is1[:, 11][xs_is1[:, 11]>0].apply_(add_phase)

xs_is2 = get_internal_coordinates_from_trajectory(traj_is2, get_construction_table(), temperature=300)

save_internal_coordinates_to_csv(xs_is1, get_construction_table(), filename='is1_lcao_zmat.csv')
save_internal_coordinates_to_csv(xs_is2, get_construction_table(), filename='is2_lcao_zmat.csv')