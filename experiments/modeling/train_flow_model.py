import torch

from flonacomldft.utils.io_utils import load_csv_file
from flonacomldft.internal_coordinates import Coordinates_mapping
from flonacomldft.models.real_nvp import RealNVP_MLP
from flonacomldft.train_flow import train_flow

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# for flows

n_iter = 8000
lr = 1e-3
mode_label = 0 # or 1

torch.manual_seed(100)

# load data

mode_label = 0 #1

zmat_train = load_csv_file("datasets/is{:d}_md_train.csv".format(mode_label))
zmat_test = load_csv_file("datasets/is{:d}_md_test.csv".format(mode_label))

# real centered frame

coord_mapping = Coordinates_mapping()
xs_train, logdetjacs_train, energies_train = coord_mapping.get_real_centered_from_internal(
                                    zmat_train[:, :12],
                                    zmat_train[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_train[:, 13]
                                    )

xs_test, logdetjacs_test, energies_test = coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 12],
                                    isomer=mode_label,
                                    energies=zmat_test[:, 13]
                                    )


model = RealNVP_MLP(12,
                    n_blocks=12, #12,
                    block_depth=1,
                    init_weight_scale=1e-3,
                    hidden_dim=16,  #128 #32
                    hidden_depth=4,  #4   #8
                    device=device,
                    )

_ = train_flow(
    model,
    xs_train.float(),
    xs_test.float(),
    isomer=mode_label,
    n_iter=n_iter,
    lr=lr,
    use_scheduler=False,
    step_schedule=100,
    save_splits=10,
    grad_clip=1e4,
)

xs_sample = _['model'].sample(1)

print(xs_sample)

zmat_sample, logdetjac_sample = coord_mapping.get_internal_from_real_centered(xs_sample, isomer=mode_label)

print(xs_sample[0] + coord_mapping.zmat_minima[mode_label])

print(zmat_sample[0])

xyz = coord_mapping.get_cartesian_from_internal(zmat_sample[0])
print(xyz)

from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
plot_atoms(xyz[0].get_ase_atoms())
plt.show()

