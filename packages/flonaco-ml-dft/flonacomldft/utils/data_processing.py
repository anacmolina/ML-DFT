import torch
import pandas as pd

# split data

def split_data_from_dataframe(tensor, train_size, sk_seed):
    
    import sklearn.model_selection

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
def centering_in_radian(xs, x_rad_center=None, return_centering_args=True):

    from flonacomldft.models.real_nvp import Angles_mapping

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


 