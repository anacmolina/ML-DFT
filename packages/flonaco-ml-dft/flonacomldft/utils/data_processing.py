import torch
import pandas as pd
import sklearn.model_selection
from flonacomldft.models.real_nvp import Angles_mapping

# split data

def split_data_from_dataframe(tensor, train_size, sk_seed):

    arrays = [tensor.numpy()]

    x_train, x_test = sklearn.model_selection.train_test_split(*arrays,
                                                      test_size=None,
                                                      train_size=train_size,
                                                      random_state=sk_seed,
                                                      shuffle=True,
                                                      stratify=None)

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    return x_train, x_test 

# centering data in radians
# TODO: look into the centering args convention
def centering_in_radian(xs, x_rad_center=None, return_centering_args=True):

    # centering in radian
    if x_rad_center is None:
        x_rad_center = xs.mean(dim=0)
    else:
        x_rad_center=x_rad_center

    x_rad_centered = xs - x_rad_center

    # computing the tanh in order to estimate the covariance for the flow base distribution 
    x_real_centered, _ = Angles_mapping().rads_to_reals(x_rad_centered)
    
    if return_centering_args:
        cov_real = torch.cov(x_real_centered.T)
        centering_args = {"cov_base": cov_real, "mean_out": x_rad_center}

        return x_real_centered, centering_args
    else:
        return x_real_centered


 # TODO: delete get mix, add new function
def get_pos_energy(zmat):
     u_tensor = zmat[:, -1]
     x_tensor = zmat[:, :-1]
     return x_tensor, u_tensor

def shuffle_arr(vs, indexes):
    concat = lambda vs: torch.cat(vs)
    v = concat(vs)
    return v[indexes]

def get_mix_data(data_1, data_2):
    xi_is1, ui_is1 = get_pos_energy(data_1)
    xi_is2, ui_is2 = get_pos_energy(data_2)
    ci_is1, ci_is2 = torch.zeros(xi_is1.shape[0]), torch.ones(xi_is2.shape[0]) 

    n_points = xi_is1.shape[0] + xi_is2.shape[0]

    indexes = torch.randperm(n_points)

    # Unifying all data from MD and shuffling in order to to Metropolis-Hastings
    xis = shuffle_arr([xi_is1, xi_is2], indexes)
    uis = shuffle_arr([ui_is1, ui_is2], indexes)
    cis = shuffle_arr([ci_is1, ci_is2], indexes)
    return xis, uis, cis