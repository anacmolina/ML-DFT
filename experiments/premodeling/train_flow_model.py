import torch

from flonacomldft.utils.io_utils import (
    load_csv_file,
    save_pickle_file
)

from flonacomldft.utils.data_processing import (
    split_data_from_dataframe,
    centering_in_radian
)

from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow_from_data import train_flow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for flows


n_iter = 5000 #100 # long_training 10000
lr = 1e-4
mode_label = 1 # or 2

torch.manual_seed(100)

# load data

xs = load_csv_file("is{:d}_lcao_zmat.csv".format(mode_label))
xs = xs[:, :-2] # remove energy and logdetjac values

train_size = 0.8
sk_seed = 42

xs_train, xs_test = split_data_from_dataframe(xs, train_size, sk_seed)

xs_train_mean = xs_train.mean(dim=0)

xs_train, centering_args = centering_in_radian(xs_train)
xs_test = centering_in_radian(xs_test, xs_train_mean, return_centering_args=False)

model = RealNVP_MLP(12,
                    n_blocks=8,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    centering_args=centering_args,
                    device=device,
                    )

out = train_flow(
    model,
    xs_train,
    xs_test,
    n_iter=n_iter,
    lr=lr,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

f = "is{:d}_flow_dic_training_long_training.pkl".format(mode_label)
save_pickle_file(out, f)

