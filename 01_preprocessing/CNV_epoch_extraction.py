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


new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants/bad_interpolated/Epochs/CNV'
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

Conditions = ['One', 'Two']

CNV_epochs_df = pd.DataFrame()

for id, cond in product(participant_ids, Conditions):
    epochs = mne.read_epochs(
        f'{id}_CNV_{cond}-epo.fif', preload=True)

    temp_df = epochs.to_data_frame()
    temp_df['Participant'] = id
    temp_df['Condition'] = cond
    CNV_epochs_df = pd.concat([CNV_epochs_df, temp_df], ignore_index=True)

    print('##########################################################')
    print('##########################################################')
    print(f'{id}_{cond} done')
    print('##########################################################')
    print('##########################################################')    

# Save the DataFrame to a CSV file
CNV_epochs_df.to_csv('../../../../ML/CNV_epochs_df.csv', index=False)