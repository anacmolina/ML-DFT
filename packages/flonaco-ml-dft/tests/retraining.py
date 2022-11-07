from flonacomldft.train_mlp_from_data import train_mlp

from flonacomldft.data_utils import (
    get_path,
    load_from_pickle,
    save_pickle_file
    
)

import torch
import pandas as pd

#MLP retraining, watch loss.backward(retain_graph=True)

data = torch.tensor(pd.read_csv(get_path() + 'x_nf_is1.csv').to_numpy())
model = load_from_pickle('mlp_is1')

model = model['mlp_model']

x = data[:, :-1]
y = data[:, -1]

print(x, y)

_ = train_mlp(model, x, y, retraining=False)

save_pickle_file(_, 'mlp_is1_retraining')
