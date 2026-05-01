# region setting up the environment
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import pandas as pd
from matplotlib.cm import ScalarMappable
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from mne.minimum_norm import apply_inverse, make_inverse_operator
import pprint
from collections import Counter
from mne.time_frequency import tfr_morlet

from mpl_toolkits.axes_grid1 import make_axes_locatable



new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")
# endregion setting up the environment

###############################################################################
# region data wrangling

participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
                    'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21', 'P23', 'P24',
                    'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']


tmin_list = np.arange(0, 2, 0.25).tolist()
tmax_list = np.arange(0.25, 2.25, 0.25).tolist()


# Creating Evoked Objects 
Epochs_One = []

for pid in participant_ids:
    path = os.path.join('bad_interpolated/Epochs/CNV', f'{pid}_CNV_One-epo.fif')
    epochs = mne.read_epochs(path, preload=True)

    # optionally restrict to event 96 now:
    epochs_96 = epochs['96']

    Epochs_One.append(epochs_96)

Evoked_One = mne.concatenate_epochs(Epochs_One).average()

Epochs_Two = []

for pid in participant_ids:
    path = os.path.join('bad_interpolated/Epochs/CNV', f'{pid}_CNV_Two-epo.fif')
    epochs = mne.read_epochs(path, preload=True)

    # optionally restrict to event 96 now:
    epochs_96 = epochs['96']

    Epochs_Two.append(epochs_96)

Evoked_Two = mne.concatenate_epochs(Epochs_Two).average()

# endregion data wrangling
###############################################################################
###############################################################################

# region topo plots

for t_min, t_max in zip(tmin_list, tmax_list):

    time_mask = (Evoked_One.times >= t_min) & (Evoked_One.times <= t_max)
    mean_data = Evoked_One.data[:, time_mask].mean(axis=1)
    fig = mne.viz.plot_topomap(
        mean_data,
        Evoked_One.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        show=False
    )
    plt.savefig(f'../ML/figs/topomaps/One/evoked_one_topomap_{t_min}_{t_max}.png', dpi=300)



for t_min, t_max in zip(tmin_list, tmax_list):

    time_mask = (Evoked_Two.times >= t_min) & (Evoked_Two.times <= t_max)
    mean_data = Evoked_Two.data[:, time_mask].mean(axis=1)
    fig = mne.viz.plot_topomap(
        mean_data,
        Evoked_Two.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        show=False
    )
    plt.savefig(f'../ML/figs/topomaps/Two/evoked_two_topomap_{t_min}_{t_max}.png', dpi=300)


freqs = np.arange(0.5, 40.5, 0.5)  # Frequency range
n_cycles = freqs / 2.0  # Number of cycles for each frequency
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']


# Extracting power 

Power_One = Evoked_One.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles
    )

Power_Two = Evoked_Two.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles
    )

# Alpha
# Averaged power values across time windows

# One
rows = []

for t_min, t_max in zip(tmin_list, tmax_list):

    fmin, fmax = 8, 13
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })

Alpha_One_values = pd.DataFrame(rows)

Alpha_One_min = Alpha_One_values['min'].min()
Alpha_One_max = Alpha_One_values['max'].max()

# creating the colorbar

# Create a figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(6, 1))

# Create a normalizer with your min/max range
norm = mcolors.Normalize(vmin=Alpha_One_min, vmax=Alpha_One_max)

# Create a ScalarMappable with your colormap
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)

# Create the colorbar
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)

plt.tight_layout()
plt.savefig('../ML/figs/topomaps/One/power/Alpha/alpha_one_colorbar.png', dpi=300)
plt.close()



# Two
rows = []

for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 8, 13
    time_mask = (Power_Two.times >= t_min) & (Power_Two.times <= t_max)
    freq_mask = (Power_Two.freqs >= fmin) & (Power_Two.freqs <= fmax)

    band_data = Power_Two.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })

Alpha_Two_values = pd.DataFrame(rows)

Alpha_Two_min = Alpha_Two_values['min'].min()
Alpha_Two_max = Alpha_Two_values['max'].max()


# Create a figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(6, 1))

# Create a normalizer with your min/max range
norm = mcolors.Normalize(vmin=Alpha_Two_min, vmax=Alpha_Two_max)

# Create a ScalarMappable with your colormap
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)

# Create the colorbar
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)

plt.tight_layout()
plt.savefig('../ML/figs/topomaps/Two/power/Alpha/alpha_two_colorbar.png', dpi=300)
plt.close()


Alpha_pwr_values = pd.concat([Alpha_One_values, Alpha_Two_values], axis=0, ignore_index=True)

# Creating power topomaps
# One
for t_min, t_max in zip(tmin_list, tmax_list):

    fmin, fmax = 8, 13
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_One.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (Alpha_min, Alpha_max),
        show=False
    )


    plt.savefig(f'../ML/figs/topomaps/One/power/Alpha/pwr_one_{t_min}_{t_max}.png', dpi=300)

# Two
for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 8, 13
    time_mask = (Power_Two.times >= t_min) & (Power_Two.times <= t_max)
    freq_mask = (Power_Two.freqs >= fmin) & (Power_Two.freqs <= fmax)

    band_data = Power_Two.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_Two.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (Alpha_min, Alpha_max),
        show=False
    )

    plt.savefig(f'../ML/figs/topomaps/Two/power/Alpha/pwr_two_{t_min}_{t_max}.png', dpi=300)

# Beta
# Averaged power values across time windows
rows = []
for t_min, t_max in zip(tmin_list, tmax_list):

    fmin, fmax = 13, 30
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })

Beta_One_values = pd.DataFrame(rows)

Beta_One_min = Beta_One_values['min'].min()
Beta_One_max = Beta_One_values['max'].max()

# colour bar for Beta One
fig, ax = plt.subplots(figsize=(6, 1))
norm = mcolors.Normalize(vmin=Beta_One_min, vmax=Beta_One_max)
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)
plt.tight_layout()
plt.savefig('../ML/figs/topomaps/One/power/Beta/beta_one_colorbar.png', dpi=300)
plt.close()

rows = []
for t_min, t_max in zip(tmin_list, tmax_list):

    fmin, fmax = 13, 30
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })

Beta_Two_values = pd.DataFrame(rows)

Beta_Two_min = Beta_Two_values['min'].min()
Beta_Two_max = Beta_Two_values['max'].max()

Beta_pwr_values = pd.concat([Beta_One_values, Beta_Two_values], axis=0, ignore_index=True)

# colour bar for Beta Two
fig, ax = plt.subplots(figsize=(6, 1))
norm = mcolors.Normalize(vmin=Beta_Two_min, vmax=Beta_Two_max)
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)
plt.tight_layout()
plt.savefig('../ML/figs/topomaps/Two/power/Beta/beta_two_colorbar.png', dpi=300)
plt.close()


# Creating power topomaps

for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 13, 30
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_One.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (Beta_min, Beta_max),
        show=False
    )

    plt.savefig(f'../ML/figs/topomaps/One/power/Beta/pwr_One_{t_min}_{t_max}.png', dpi=300)   

for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 13, 30
    time_mask = (Power_Two.times >= t_min) & (Power_Two.times <= t_max)
    freq_mask = (Power_Two.freqs >= fmin) & (Power_Two.freqs <= fmax)

    band_data = Power_Two.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_Two.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (Beta_min, Beta_max),
        show=False
    )

    plt.savefig(f'../ML/figs/topomaps/Two/power/Beta/pwr_Two_{t_min}_{t_max}.png', dpi=300)


# Theta
# Averaged power values across time windows
# One 
rows = []
for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 4, 8
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })

Theta_One_values = pd.DataFrame(rows)

Theta_One_min = Theta_One_values['min'].min()
Theta_One_max = Theta_One_values['max'].max()

# colour bar for Theta One
fig, ax = plt.subplots(figsize=(6, 1))
norm = mcolors.Normalize(vmin=Theta_One_min, vmax=Theta_One_max)
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)
plt.tight_layout()
plt.savefig('../ML/figs/topomaps/One/power/Theta/theta_one_colorbar.png', dpi=300)
plt.close()

# Two
rows = []
for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 4, 8
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    rows.append({
        "min": band_data.min(),
        "max": band_data.max()
    })
Theta_Two_values = pd.DataFrame(rows)

Theta_Two_min = Theta_Two_values['min'].min()
Theta_Two_max = Theta_Two_values['max'].max()

Theta_pwr_values = pd.concat([Theta_One_values, Theta_Two_values], axis=0, ignore_index=True)


# colour bar for Theta Two
fig, ax = plt.subplots(figsize=(6, 1))
norm = mcolors.Normalize(vmin=Theta_Two_min, vmax=Theta_Two_max)
cmap = plt.cm.RdBu_r  # Use the same colormap as your topomap
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
cbar.set_label('Power (dB)', fontsize=12)
plt.tight_layout()
plt.savefig('../ML/figs/topomaps/Two/power/Theta/theta_two_colorbar.png', dpi=300)
plt.close()



# Creating power topomaps
# One
for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 4, 8
    time_mask = (Power_One.times >= t_min) & (Power_One.times <= t_max)
    freq_mask = (Power_One.freqs >= fmin) & (Power_One.freqs <= fmax)

    band_data = Power_One.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_One.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (Theta_min, Theta_max),
        show=False
    )

    plt.savefig(f'../ML/figs/topomaps/One/power/Theta/pwr_one_topomap_{t_min}_{t_max}.png', dpi=300)   

for t_min, t_max in zip(tmin_list, tmax_list):
    fmin, fmax = 4, 8
    time_mask = (Power_Two.times >= t_min) & (Power_Two.times <= t_max)
    freq_mask = (Power_Two.freqs >= fmin) & (Power_Two.freqs <= fmax)

    band_data = Power_Two.data[:, freq_mask][:, :, time_mask].mean(axis=(1, 2))

    fix, ax = mne.viz.plot_topomap(
        band_data,
        Power_Two.info,
        contours=6,
        cmap='RdBu_r',
        size=3,
        #vlim = (overall_min, overall_max),
        show=False
    )

    plt.savefig(f'../ML/figs/topomaps/Two/power/Theta/pwr_two_topomap_{t_min}_{t_max}.png', dpi=300)

plt.close()

# combined min & max values 

Combined_values = pd.concat([
    Alpha_pwr_values, Beta_pwr_values, Theta_pwr_values], 
    axis=0, ignore_index=True)

overall_min = Combined_values['min'].min()
overall_max = Combined_values['max'].max()

# endregion topo plots

###############################################################################

# region LORETA
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from mne.minimum_norm import apply_inverse, make_inverse_operator

Conditions = ['One', 'Two']

src = mne.read_source_spaces('fsaverage-src.fif')
Bem = mne.read_bem_solution('fsaverage-bem-sol.fif')
trans = 'fsaverage-trans.fif'

fsaverage_dir = os.path.expanduser(new_directory)
os.environ['SUBJECTS_DIR'] = fsaverage_dir
subject = 'fsaverage'


# Combined Noise Covariance:
One_noise_cov = mne.compute_covariance(
    Epochs_One, tmin= -0.1, tmax=0)

Two_noise_cov = mne.compute_covariance(
    Epochs_Two, tmin= -0.1, tmax=0)


# Importing fwd sols:
for cond in Conditions:
    globals()[f'{cond}_fwd_solution'] = mne.read_forward_solution(
        os.path.join('bad_interpolated/fwd_sols', f'{cond}_CNV_fwd_solution-fwd.fif'))
    

for cond in Conditions:
    globals()[f'{cond}_inv_op'] = make_inverse_operator(
        (globals()[f'Evoked_{cond}']).info, 
        globals()[f'{cond}_fwd_solution'], 
        globals()[f'{cond}_noise_cov'])
    print(f"Inverse operator for pooled condition {cond} created.")


# Creating inverse solution:
method = 'eLORETA'
snr = 4.
lambda2 = 1. / snr ** 2


for cond in Conditions:
    globals()[f'{cond}_invs_sol'] = apply_inverse(
        globals()[f'Evoked_{cond}'], 
        globals()[f'{cond}_inv_op'], 
        lambda2, method=method, pick_ori=None)
    print(f"Inverse solution for pooled condition {cond} created.")



for cond in Conditions:
    stc = globals()[f'{cond}_invs_sol']  # original (don’t mutate)

    for t_min, t_max in zip(tmin_list, tmax_list):
        print(f'Plotting LORETA for condition {cond} from {t_min} to {t_max} seconds.')

        mean_data = stc.copy().crop(tmin=t_min, tmax=t_max).mean()

        brain = mean_data.plot(
            hemi='both',
            views='dorsal',
            time_viewer=False,
            smoothing_steps=10,
            background='white',
            size=(1000, 600),
            colorbar=True,
            clim='auto'
        )

        brain._renderer.plotter.camera.zoom(0.8)
        
        brain._renderer.plotter.render()

        brain.save_image(f'../ML/figs/LORETA/{cond}/loreta_{t_min}_{t_max}.png')

        brain.close()



temp_stc = One_invs_sol.copy().crop(tmin=0, tmax=0.25).mean()
brain = temp_stc.plot(
    hemi='both',
    views='dorsal',
    time_viewer=False,
    smoothing_steps=10,
    background='white',
    size=(1000, 600),
    colorbar=True,
    clim='auto'
)

brain._renderer.plotter.camera.zoom(0.8)
