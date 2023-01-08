import torch

from flonacomldft.utils.io_utils import load_csv_file
from flonacomldft.utils.data_processing import split_data_from_dataframe

from flonacomldft.models.mlp import MLP
from flonacomldft.train_mlp_from_data import train_mlp

from flonacomldft.utils.io_utils import save_pickle_file
import matplotlib.pyplot as plt

mode_label = 2 #or 2

xs_md = load_csv_file("is{:d}_lcao_zmat.csv".format(mode_label))
xs_flow = load_csv_file("is{:d}_flow_zmat.csv".format(mode_label))

train_size = 0.8
sk_seed = 42

x_md_train, x_md_test = split_data_from_dataframe(xs_md, train_size, sk_seed)
x_flow_train, x_flow_test = split_data_from_dataframe(xs_flow, train_size, sk_seed)

# concat of MD configs and random generated samples from flows
train = torch.cat((x_md_train, x_flow_train))
test = torch.cat((x_md_test, x_flow_test))

# remove logdetcat and split input (internal coordinates) and output (energies)
x_train = train.clone()[:, :-2]
x_test = test.clone()[:, :-2]
u_train = train.clone()[:, -1]
u_test = test.clone()[:, -1]

n_hidden = 64
n_layers = 2
model = MLP([x_train.shape[1]] +  [n_hidden] * n_layers + [1])

mlp_hyperparams = {'n_iter': 5000,
    'lr': 5e-4,
    'use_scheduler': False,
    'step_schedule': 100,
}

out = train_mlp(model, x_train, x_test, u_train, u_test, **mlp_hyperparams, 
              with_tqdm=True)

plt.figure()
plt.plot(out['losses'][0], label='train')
plt.plot(out['losses'][1], label='test')
plt.legend()
plt.show(block=False)

f = "is{:d}_mlp_dic_training.pkl".format(mode_label)
save_pickle_file(out, f)