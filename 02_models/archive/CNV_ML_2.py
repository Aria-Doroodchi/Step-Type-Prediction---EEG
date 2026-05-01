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

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")
# endregion setting up the environment

###############################################################################
# region data wrangling: 

# global variables
participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
                    'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P21', 'P23', 'P24',
                    'P25', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']

conditions = ['One', 'Two']
bin_n = 1/8
freqs = np.arange(0.5, 40.5, 0.5)  # Frequency range
n_cycles = freqs / 2.0  # Number of cycles for each frequency
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

test_size_n = 0.40  # 40% test size

# epoch info for data wrangling
P01_One_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_One-epo.fif')
ch_names_list = P01_One_epochs.ch_names.copy()
ch_names_list.remove('Stim')

# data frame for storing model performance results
summary_df = pd.DataFrame()

for id in participant_ids:
    per_participant = []

    for cond in conditions: 
        path = os.path.join('bad_interpolated/Epochs/CNV', f'{id}_CNV_{cond}-epo.fif')
        epochs = mne.read_epochs(path, preload=True)
        epochs = epochs.crop(tmin=0, tmax=2.0)

        # region epochs 
        epoch_df = epochs.to_data_frame()
        epoch_df['bin'] = (epoch_df['time'] // bin_n).astype(int)
        epoch_df = epoch_df.drop(columns=['Stim', 'condition', 'time'])

        epoch_df = (epoch_df
                    .groupby(['epoch', 'bin'])
                    .mean(numeric_only=True)
                    .reset_index()
        )

        epoch_long = epoch_df.melt(
            id_vars=['epoch', 'bin'],
            value_vars=ch_names_list,
            var_name='channel',
            value_name='amplitude'
        )

        epoch_wide = (
            epoch_long
            .pivot(
                index=['epoch'],
                columns=['channel', 'bin'],
                values='amplitude'
            )
            .reset_index()
        )

        epoch_wide.columns = [
            f'{ch}_bin_{b}' for ch, b in epoch_wide.columns
        ]
        epoch_wide = epoch_wide.rename(
            columns={'epoch_bin_': 'epoch'}
        )

        print(f'epochs data frame for {id} {cond} created:')
        print(epoch_wide.head())

        #region PSD

        power = epochs.compute_tfr(
            method='morlet',
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=False
        )

        power_df = power.to_data_frame()
        freq_bands = [
            power_df['freq'].between(0.5, 4.0),
            power_df['freq'].between(4.0, 8.0),
            power_df['freq'].between(8.0, 13.0),
            power_df['freq'].between(13.0, 30.0),
            power_df['freq'].between(30.0, 40.0)
        ]

        power_df['freq'] = np.select(freq_bands, freq_band_names, default=power_df['freq'])

        power_df['bin'] = (power_df['time'] // bin_n).astype(int)
        power_df = power_df.drop(columns=['condition', 'time'])

        power_df_avg = (power_df
                        .groupby(['freq', 'epoch', 'bin'])
                        .mean(numeric_only=True)
                        .reset_index()
                        )
        
        power_df_avg_long = power_df_avg.melt(
            id_vars = ['freq', 'epoch', 'bin'],
            value_vars = ch_names_list,
            var_name = 'channel',
            value_name = 'power'
            )
        
        power_df_avg_wide = (
            power_df_avg_long
            .pivot(
                index = ['epoch', 'bin'],
                columns = ['channel', 'freq'],
                values = 'power'
                )
            .reset_index()
            )
        
        power_df_avg_wide.columns = [
            f'{ch}_{band}' for ch, band in power_df_avg_wide.columns]
        
        power_df_avg_wide = power_df_avg_wide.rename(
            columns = {'epoch_': 'epoch',
                       'bin_': 'bin'})
        

        power_df_binned = power_df_avg_wide.melt(
                id_vars = ['epoch', 'bin'],
                value_vars = power_df_avg_wide.columns[2:],
                var_name = 'channel_freqband',
                value_name = 'power'
        )

        power_df_binned_wide = (power_df_binned
                                .pivot(
                                    index = ['epoch'],
                                    columns = ['channel_freqband', 'bin'],
                                    values = 'power'
                                )
                                .reset_index()
        )

        power_df_binned_wide.columns = [
            f'{chfb}_bin_{b}' for chfb, b in power_df_binned_wide.columns
        ]

        power_df_binned_wide = power_df_binned_wide.rename(
            columns={'epoch_bin_': 'epoch'})
        
        print(f'PSD data frame for {id} {cond} created:')
        print(power_df_binned_wide.head())

        
        merged_df = pd.merge(
        epoch_wide,
        power_df_binned_wide,
        on = ['epoch'])

        merged_df['condition'] = cond
        merged_df['participant_id'] = id

        per_participant.append(merged_df)

    final_df = pd.concat(per_participant, ignore_index=True)

    df = final_df.drop(columns=['participant_id'])

    print(f'Final data frame for {id} created:')
    print(df.head())

    # region ML

    # 1) Split features/target and encode labels
    X = df.drop(columns=['condition'])
    y = df['condition'].map({'One': 0, 'Two': 1}).astype(int)

    # 2) Train/val/test split (stratified to keep class balance)
    X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=test_size_n, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    # 3) Scale features (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4) Handle class imbalance (optional but recommended)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train)

    class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}

    
    # 5) Build a simple, strong baseline MLP
    def build_model(input_dim: int) -> tf.keras.Model:
        model = Sequential([
            Input(shape=(input_dim,)),
            BatchNormalization(),
            Dense(1024, activation='relu'),
            Dropout(0.30),
            Dense(512, activation='relu'),
            Dropout(0.20),
            Dense(1, activation='sigmoid')  # binary output
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        return model

    model = build_model(X_train_s.shape[1])

    # 6) Train with early stopping on validation AUC
    early_stop = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=15,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 7) Evaluate on test set
    proba_test = model.predict(X_test_s).ravel()
    pred_test  = (proba_test >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_one = tn + fp 
    total_two = fn + tp
    correct_one = tn
    correct_two = tp

    acc_one = correct_one / total_one if total_one else float("nan")
    acc_two = correct_two / total_two if total_two else float("nan")

    temp_summary_df = pd.DataFrame({
        "Participant_ID": [id],
        "total_One": [total_one],
        "total_Two": [total_two],
        "correct_One": [correct_one],
        "correct_Two": [correct_two],
        "Accuracy_One": [acc_one],
        "Accuracy_Two": [acc_two]
    })

    summary_df = pd.concat([summary_df, temp_summary_df], ignore_index=True)

    print(f'{id} processed. One accuray: {acc_one:.3f}, Two accuracy: {acc_two:.3f}')

total_One = summary_df['total_One'].sum()
total_Two = summary_df['total_Two'].sum()
correct_One = summary_df['correct_One'].sum()
correct_Two = summary_df['correct_Two'].sum()
overall_acc_one = correct_One / total_One if total_One else float("nan")
overall_acc_two = correct_Two / total_Two if total_Two else float("nan")

total_accuracy = (correct_One + correct_Two) / (total_One + total_Two)

overall_acc_one
overall_acc_two
total_accuracy