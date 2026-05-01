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
import sklearn

new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")
# endregion setting up the environment

###############################################################################
# region data wrangling: 

participant_ids = ['P01', 'P02']
bin_n = 1/8

P01_One_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_One-epo.fif')
P01_One_epochs = P01_One_epochs.crop(tmin=0, tmax=2)

ch_names_list = P01_One_epochs.ch_names.copy()
ch_names_list.remove('Stim')

P01_One_epochs_df = P01_One_epochs.to_data_frame()

P01_One_epochs_df['bin'] = (P01_One_epochs_df['time'] // bin_n).astype(int)

# Calculate slopes for each epoch, bin, and channel
P01_One_epochs_slopes_df = P01_One_epochs_df.copy().drop(columns=['Stim', 'condition'])

P01_One_epochs_slopes_df = (P01_One_epochs_slopes_df
    .groupby(['epoch', 'bin'])
    .apply(lambda x: x[ch_names_list].apply(lambda y: np.polyfit(x['time'], y, 1)[0]), 
           include_groups=False)
    .reset_index()
)

threshold_constant = 0.9

for col in P01_One_epochs_slopes_df.columns:
    top_freq = P01_One_epochs_slopes_df[col].value_counts(normalize=True, dropna=False).iloc[0]
    if top_freq > threshold_constant:
        P01_One_epochs_slopes_df = P01_One_epochs_slopes_df.drop(columns=[col])




P01_One_epochs_slopes_df_long = P01_One_epochs_slopes_df.melt(
    id_vars = ['epoch', 'bin'],
    value_vars = ch_names_list,
    var_name = 'channel',
    value_name = 'slope'
)

P01_One_epochs_slopes_df_wide = (
    P01_One_epochs_slopes_df_long
    .pivot(
        index=['epoch'],
        columns=['channel', 'bin'],
        values='slope'
    )
    .reset_index()
)

P01_One_epochs_slopes_df_wide.columns = [
    f'slope_{ch}_bin_{b}' for ch, b in P01_One_epochs_slopes_df_wide.columns]

P01_One_epochs_slopes_df_wide = P01_One_epochs_slopes_df_wide.rename(
    columns={'slope_epoch_bin_': 'epoch'})


# Calculate mean amplitudes for each epoch, bin, and channel
P01_One_epochs_df = P01_One_epochs_df.drop(columns=['Stim', 'condition', 'time'])

P01_One_epochs_df = (P01_One_epochs_df
    .groupby(['epoch', 'bin'])
    .mean(numeric_only=True)
    .reset_index()
)

P01_One_epochs_df_long = P01_One_epochs_df.melt(
    id_vars = ['epoch', 'bin'],
    value_vars = ch_names_list,
    var_name = 'channel',
    value_name = 'amplitude'
)


P01_One_epochs_df_wide = (
    P01_One_epochs_df_long
    .pivot(
        index=['epoch'],
        columns=['channel', 'bin'],
        values='amplitude'
    )
    .reset_index()
)

P01_One_epochs_df_wide.columns = [f'{ch}_bin_{b}' for ch, b in P01_One_epochs_df_wide.columns]
P01_One_epochs_df_wide = P01_One_epochs_df_wide.rename(
    columns={'epoch_bin_': 'epoch'})

df = P01_One_epochs_df_wide.merge(
    P01_One_epochs_slopes_df_wide,
    on='epoch'
)

###############################################################################
# region PSD 
# Compute time-frequency representation with 0.5s windows
# Using multitaper method for better frequency resolution
freqs = np.arange(0.5, 40.5, 0.5)  # Frequency range
n_cycles = freqs / 2.0  # Number of cycles for each frequency

power_One = P01_One_epochs.compute_tfr(
    method='morlet',
    freqs=freqs,
    n_cycles=n_cycles,
    return_itc=False,
    average=False  # Keep individual epochs
)

power_One_df = power_One.to_data_frame()
# Define frequency bands
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

freq_bands_One = [
    power_One_df['freq'].between(0.5, 4.0),  # Delta
    power_One_df['freq'].between(4.0, 8.0),  # Theta
    power_One_df['freq'].between(8.0, 13.0),  # Alpha
    power_One_df['freq'].between(13.0, 30.0),  # Beta
    power_One_df['freq'].between(30.0, 40.0)   # Gamma
]

power_One_df['freq'] = np.select(freq_bands_One, freq_band_names, default=power_One_df['freq'])

power_One_df['bin'] = (power_One_df['time'] // bin_n).astype(int)
power_One_df = power_One_df.drop(columns=['time', 'condition'])

power_One_df_avg = (power_One_df
    .groupby(['freq', 'epoch', 'bin'])
    .mean(numeric_only=True)
    .reset_index()
)

power_One_df_long = power_One_df_avg.melt(
    id_vars = ['freq', 'epoch', 'bin'],
    value_vars = ch_names_list,
    var_name = 'channel',
    value_name = 'power'
)


power_df_One_wide = (
    power_One_df_long
    .pivot(index=['epoch', 'bin'],
              columns=['channel', 'freq'], 
              values='power')
    .reset_index()
)
power_df_One_wide.columns = [f'{ch}_{band}' for ch, band in power_df_One_wide.columns]

power_df_One_wide = power_df_One_wide.rename(
    columns={'epoch_': 'epoch',
              'bin_': 'bin'})

power_df_binned_One = power_df_One_wide.melt(
    id_vars = ['epoch', 'bin'],
    value_vars = power_df_One_wide.columns[2:],
    var_name = 'channel_freqband',
    value_name = 'power'
)

power_df_One_binned_long = (
    power_df_binned_One
    .pivot(
        index=['epoch'],
        columns=['channel_freqband', 'bin'],
        values='power')
    .reset_index()
)

power_df_One_binned_long.columns = [f'{chfb}_bin_{b}' for chfb, b in power_df_One_binned_long.columns]

power_df_One_binned_wide = power_df_One_binned_long.rename(
    columns={'epoch_bin_': 'epoch'})
# merge dfs: 

df = (P01_One_epochs_slopes_df_wide
      .merge(P01_One_epochs_df_wide, on='epoch')
      .merge(power_df_One_binned_wide, on='epoch')
)

###############################################################################
# region feature selection

# Example threshold (tune as needed)
threshold = 0.9  # remove one of any pair of columns with |correlation| > 0.9

# Compute correlation matrix (numeric columns only)
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find columns with correlation above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop them
df_reduced = df.drop(columns=to_drop)

print(f"Removed {len(to_drop)} highly correlated columns.")

# Univariate tests

