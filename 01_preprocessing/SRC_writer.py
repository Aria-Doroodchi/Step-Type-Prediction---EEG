# region setting up the environment
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from mne.minimum_norm import apply_inverse, make_inverse_operator
import pprint
from collections import Counter
from itertools import product


new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")



# endregion setting up the environment
###############################################################################
###############################################################################
# region Loading the data
participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
                    'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21', 'P23', 'P24',
                    'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']


conditions = ['One', 'Two']

min_time_var = 0 #onset time for predictions. must be betweeen 0 and 2
bin_n = 1/8 # over a 2 second window 

src = mne.read_source_spaces('fsaverage-src.fif')
Bem = mne.read_bem_solution('fsaverage-bem-sol.fif')
trans = 'fsaverage-trans.fif'
method = 'eLORETA'
snr = 2.
lambda2 = 1. / snr ** 2

Data_path = mne.datasets.sample.data_path()
Subjects_dir = Data_path / "subjects"
labels = mne.read_labels_from_annot('fsaverage',
                                    parc = 'aparc.a2009s',
                                    subjects_dir = Subjects_dir)


ba_names = [label.name for label in labels]


for id in participant_ids:
    per_participant = []

        # region data wrangling 

    #importing epoch data 
    for cond in conditions: 
        path = os.path.join('bad_interpolated/Epochs/CNV', f'{id}_CNV_{cond}-epo.fif')
        epochs = mne.read_epochs(path, preload=True)

        # region source localization 

        LORETA_epoch_rows = []

        epoch_nums = epochs.selection.tolist()

        for idx, num in enumerate(epoch_nums):
            sub_epochs = epochs[[idx]]
            sub_evoked = sub_epochs['96'].average()



            noise_cov = mne.compute_covariance(sub_epochs, tmin= -0.1, tmax=0)
            fwd_sol = mne.make_forward_solution(
                sub_evoked.info,
                trans = 'fsaverage',
                src = src,
                bem = Bem,
                n_jobs = 18
            )

            inv_op = make_inverse_operator(
                sub_evoked.info,
                fwd_sol,
                noise_cov
            )

            inv_sol = apply_inverse(
                sub_evoked,
                inv_op,
                lambda2,
                method = method,
            )

            bm_activity = mne.extract_label_time_course(
            inv_sol, 
            labels, 
            src = src,
            mode='mean'
            )
            
            times = sub_evoked.times
            
            bm_activity_df = pd.DataFrame(
            bm_activity,
            index = ba_names,
            columns = times).T.reset_index().rename(
                columns={'index': 'time'}
            )

            bm_activity_df = bm_activity_df[bm_activity_df['time'] >= min_time_var]
            bm_activity_df['bin'] = (bm_activity_df['time'] // bin_n).astype(int)
            bm_activity_binned = bm_activity_df.groupby('bin').mean().reset_index()



            transposed_dfs = []

            for col in ba_names:
                bm_temp = bm_activity_binned[['bin', col]]
                bm_temp_transposed = bm_temp.set_index('bin').T
                bm_temp_transposed.columns = [
                    f'{col}_bin_{b}' for b in bm_temp_transposed.columns]
                bm_temp_transposed.reset_index(drop=True, inplace=True)
                transposed_dfs.append(bm_temp_transposed)

            epoch_row = pd.concat(transposed_dfs, axis=1)
            epoch_row['epoch'] = num

            LORETA_epoch_rows.append(epoch_row)
        
        bm_df = pd.concat(LORETA_epoch_rows, axis=0, ignore_index=True)

        bm_df.to_csv(f'../ML/src/{id}_{cond}_src.csv', index=False)