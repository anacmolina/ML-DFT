import os
import time
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from abflowmc.utils.io_utils import (
    load_pickle_file,
    load_csv_file,
    set_str_date_to_int, 
    get_path,
    save_pickle_file,
    save_json_args,
    )

from abflowmc.utils.diagnostics import (
    Target_Log_Prob, 
    get_participation_ratio,
)

from abflowmc.observables.free_energy_computations import (
    compute_TFP_logratio, compute_deepBAR_logratio,
    compute_TFP, compute_BAR
)

import gpaw.mpi as mpi

ranks = np.arange(0, mpi.world.size)
rank = mpi.rank
comm = mpi.world.new_communicator(ranks)

num_seed = np.array([0])
date_start = np.array([0])

if rank == 0:
    num_seed = np.random.randint(0, 100, (1,))
    date_start = np.array([set_str_date_to_int(time.strftime('%Y-%m-%d %H:%M:%S'))])

mpi.world.barrier()

comm.broadcast(num_seed, 0)
comm.broadcast(date_start, 0)

num_seed = num_seed[0]
date_start = date_start[0]

print('Seed: ', num_seed, rank)
print('Date start: ', date_start, rank)

# define arguments to parse from command line
parser = argparse.ArgumentParser(description='Prepare experiment')
# execution params
parser.add_argument('-s', '--seed', type=int, default=num_seed)
parser.add_argument('-etype', '--energy-type', type=str, default='dft')
parser.add_argument('-prop', '--n-prop', type=int, default=50)
parser.add_argument('-samp', '--n-samp', type=int, default=10)
parser.add_argument('-skip', '--skip-frames', type=int, default=10)
parser.add_argument('-pid', '--process-id', type=int, default=date_start)
parser.add_argument('-T', '--temperature', type=float, default=350)

args = parser.parse_args()
args.date_start = str(date_start)

# set the random seed
torch.manual_seed(args.seed)

project_path = '/mnt/home/amolina/ceph/adaptive-350/adaptive-dft'

isms_files = ['1-adaptive/results_adaptive_is0_28685740/adaptive_sampling_is0_28685740.pkl',
             '1-adaptive/results_adaptive_is1_28685741/adaptive_sampling_is1_28685741.pkl']


mlps_files = ['andersen/models/dict_mlp_model_is0_adaptive.pkl', 'andersen/models/dict_mlp_model_is1_adaptive.pkl']

adaptives_dic = [load_pickle_file('/'+ f, path=project_path) for f in isms_files]
mlps_dic = [load_pickle_file('/'+ f, path=get_path()) for f in mlps_files]

folder_to_save_fe = 'results_free_energy_{:d}'.format(args.process_id)
path_fe = os.getcwd() + '/' + folder_to_save_fe
if rank == 0:
    if not os.path.exists(path_fe):
        os.makedirs(path_fe)

mode_labels = [0, 1]

n_prop = args.n_prop
n_samp = args.n_samp

skip=args.skip_frames

dic_results_TFPs = {}

for isomer_id in mode_labels:

    dic_results_TFPs[isomer_id] = {}
    dic_results_TFPs[isomer_id]['means'] = []
    dic_results_TFPs[isomer_id]['stds'] = []

    plt.figure()
    plt.title('isomer {:d}'.format(isomer_id))
    for r,run in enumerate(adaptives_dic[isomer_id]['dict_flows_training'][::skip]):
        flow = run[0]['models'][-1]
        log_ratio_0_TFPs = []
        log_err_0_TFPs = []

        #save_to_different_files

        for trial in range(n_samp):
            target_log_prob = Target_Log_Prob(energy_type=args.energy_type, 
                                    mode_label=isomer_id, 
                                    mlp_model=mlps_dic[isomer_id]['model'],
                                    T=args.temperature, 
                                    folder=path_fe+'/DFT_TPF_is{:d}_run_{:d}_trial_{:d}'.format(isomer_id, r, trial)).target_log_prob
            logr_TFP_0, log_err_TFP_0 = compute_TFP(n_prop, target_log_prob, flow)
            log_ratio_0_TFPs.append(logr_TFP_0)
            log_err_0_TFPs.append(log_err_TFP_0)
        
        dic_results_TFPs[isomer_id][r] = (log_ratio_0_TFPs, log_err_0_TFPs)
        
        plt.scatter(np.ones(len(log_ratio_0_TFPs))*r, log_ratio_0_TFPs, marker='x', alpha=0.2)
        plt.scatter(r, np.array(log_ratio_0_TFPs).mean(), c='k', marker='+')

        dic_results_TFPs[isomer_id]['means'].append(np.array(log_ratio_0_TFPs).mean())
        dic_results_TFPs[isomer_id]['stds'].append(np.array(log_ratio_0_TFPs).std())
    
    plt.errorbar(np.arange(len(dic_results_TFPs[isomer_id]['means'])), dic_results_TFPs[isomer_id]['means'], yerr=dic_results_TFPs[isomer_id]['stds'], fmt='o')
       

    plt.xlabel('adaptive mcmc retraining stage')
    plt.ylabel('log ratio by Targeted Free energy Perturbation (TFP)')

    plt.savefig(path_fe + '/TFP_isomer_{:d}.png'.format(isomer_id))

    print('logr_TFP {:.3e} +/- {:.3e}'.format(np.array(log_ratio_0_TFPs).mean(), np.array(log_ratio_0_TFPs).std()))                                    


save_pickle_file(dic_results_TFPs, 'results_TFPs.pkl', path=path_fe)

n_prop = args.n_prop
n_samp = args.n_samp

dic_results_BAR_mcmc = {}

for isomer_id in mode_labels:
    dic_results_BAR_mcmc[isomer_id] = {}
    dic_results_BAR_mcmc[isomer_id]['means'] = []
    dic_results_BAR_mcmc[isomer_id]['stds'] = []

    xs = torch.cat(adaptives_dic[isomer_id]['xs']).view(-1, 12)
    #target_log_prob = lambda x: - mlps_dic[isomer_id]['model'](x) / (kb * T)
    
    plt.figure()
    plt.title('isomer {:d}'.format(isomer_id))
    
    for r,run in enumerate(adaptives_dic[isomer_id]['dict_flows_training'][::skip]):
        flow = run[0]['models'][-1]
        log_ratio_0_BARs = []
        log_err_0_BARs = []

        for trial in range(n_samp):

            target_log_prob = Target_Log_Prob(energy_type=args.energy_type, 
                                    mode_label=isomer_id, 
                                    mlp_model=mlps_dic[isomer_id]['model'],
                                    T=args.temperature, 
                                    folder=path_fe+'/DFT_BAR_is{:d}_run_{:d}_trial_{:d}'.format(isomer_id, r, trial)).target_log_prob

            logr_BAR_0, log_err_BAR_0 = compute_BAR(xs, target_log_prob, flow, n_prop)
            log_ratio_0_BARs.append(logr_BAR_0)
            log_err_0_BARs.append(log_err_BAR_0)
        
        plt.scatter(np.ones(len(log_ratio_0_BARs))*r, log_ratio_0_BARs, marker='x', alpha=0.2)
        plt.scatter(r, np.array(log_ratio_0_BARs).mean(), c='k', marker='+')

        dic_results_BAR_mcmc[isomer_id]['means'].append(np.array(log_ratio_0_BARs).mean())
        dic_results_BAR_mcmc[isomer_id]['stds'].append(np.array(log_ratio_0_BARs).std())
    
    plt.errorbar(np.arange(len(dic_results_BAR_mcmc[isomer_id]['means'])), dic_results_BAR_mcmc[isomer_id]['means'], yerr=dic_results_BAR_mcmc[isomer_id]['stds'], fmt='o')
   
    plt.xlabel('adaptive mcmc retraining stage')
    plt.ylabel('log ratio by BAR MCMC data')

    plt.savefig(path_fe + '/BAR_isomer_{:d}.png'.format(isomer_id))

save_pickle_file(dic_results_BAR_mcmc, 'results_BARs.pkl', path=path_fe)

from abflowmc.internal_coordinates import Coordinates_mapping

def get_dataset(name, path, isomer_labels):

    coord_mapping = Coordinates_mapping()
    
    zmats = [load_csv_file("is{:d}_{:s}.csv".format(isomer_label, name), path) for isomer_label in isomer_labels] 
    xs = [coord_mapping.get_real_centered_from_internal(
                                    zmat_test[:, :12],
                                    zmat_test[:, 14],
                                    isomer=isomer_label,
                                    energies=zmat_test[:, 12]
                                    ) for isomer_label, zmat_test in zip(isomer_labels, zmats)]

    xs = [torch.cat((x[0], x[2].reshape(-1, 1), 
                                 zmat[:, 13].reshape(-1, 1), 
                                 #x[1].reshape(-1, 1),
                                 ), dim=1) for x, zmat in zip(xs, zmats)]
    
    #xs = xs.flatten(start_dim=0, end_dim=1).to(torch.float32)

    return xs, zmats

path_datasets = get_path() + '/andersen/datasets' 
dataset_labels = ['flow_train', 'flow_test']

xss_md_train, zmats_train = get_dataset('flow_train', path_datasets, mode_labels)
xss_md_test, zmats_test = get_dataset('flow_test', path_datasets, mode_labels)

dic_results_BAR_md = {}

for isomer_id in mode_labels:
    dic_results_BAR_md[isomer_id] = {}
    dic_results_BAR_md[isomer_id]['means'] = []
    dic_results_BAR_md[isomer_id]['stds'] = []

    xs = xss_md_train[isomer_id][:, :12]

    plt.figure()
    plt.title('isomer {:d}'.format(isomer_id))

    for r,run in enumerate(adaptives_dic[isomer_id]['dict_flows_training'][::skip]):
        flow = run[0]['models'][-1]
        log_ratio_0_BARs = []
        log_err_0_BARs = []

        for trial in range(n_samp):

            target_log_prob = Target_Log_Prob(energy_type=args.energy_type, 
                                    mode_label=isomer_id, 
                                    mlp_model=mlps_dic[isomer_id]['model'],
                                    T=args.temperature, 
                                    folder=path_fe+'/DFT_BAR_md_is{:d}_run_{:d}_trial_{:d}'.format(isomer_id, r, trial)).target_log_prob

            logr_BAR_0, log_err_BAR_0 = compute_BAR(xs, target_log_prob, flow, n_prop)
            log_ratio_0_BARs.append(logr_BAR_0)
            log_err_0_BARs.append(log_err_BAR_0)

        plt.scatter(np.ones(len(log_ratio_0_BARs))*r, log_ratio_0_BARs, marker='x', alpha=0.2)
        plt.scatter(r, np.array(log_ratio_0_BARs).mean(), c='k', marker='+')

        dic_results_BAR_md[isomer_id]['means'].append(np.array(log_ratio_0_BARs).mean())
        dic_results_BAR_md[isomer_id]['stds'].append(np.array(log_ratio_0_BARs).std())

    plt.errorbar(np.arange(len(dic_results_BAR_md[isomer_id]['means'])), dic_results_BAR_md[isomer_id]['means'], yerr=dic_results_BAR_md[isomer_id]['stds'], fmt='o')

    plt.xlabel('adaptive mcmc retraining stage')
    plt.ylabel('log ratio by BAR MD data')

    plt.savefig(path_fe + '/BAR_md_isomer_{:d}.png'.format(isomer_id))

save_pickle_file(dic_results_BAR_md, 'results_BARs_md.pkl', path=path_fe)

argparse_dic = vars(args)

save_json_args(args, 'free_energy', args.process_id, path_fe)

for isomer_id in mode_labels:
    plt.figure()
    plt.title('Isomer {}'.format(isomer_id))
    
    means = np.array(dic_results_BAR_mcmc[isomer_id]['means'])
    stds = np.array(dic_results_BAR_mcmc[isomer_id]['stds'])
    plt.plot(np.arange(len(means)), means, '.-', label='BAR (MCMC)')
    plt.fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.2)

    means = np.array(dic_results_BAR_md[isomer_id]['means'])
    stds = np.array(dic_results_BAR_md[isomer_id]['stds'])
    plt.plot(np.arange(len(means)), means, '.-', label='BAR (MD)')
    plt.fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.2)

    means = np.array(dic_results_TFPs[isomer_id]['means'])
    stds = np.array(dic_results_TFPs[isomer_id]['stds'])
    plt.plot(np.arange(len(means)), means, '.-', label='TFPs')
    plt.fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.2)

    plt.legend()
    plt.xlabel('Retraining stage')
    plt.ylabel('Log ratio of partition flow/MLP')

    plt.savefig(path_fe + '/comparison_isomer_{:d}.png'.format(isomer_id))
