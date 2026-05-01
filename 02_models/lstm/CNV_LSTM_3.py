import os
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import scipy

import warnings

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score

from scikeras.wrappers import KerasClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2




new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")
# endregion setting up the environment

###############################################################################
###############################################################################
# region data wrangling: 

# global variables
# data frame variables

# This script has to be ran twice because of memory limitations

# 1: 
#participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
#                    'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21']


#2: 
participant_ids = ['P23', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31',
                     'P33', 'P35', 'P37', 'P39']

###############################################################################

conditions = ['One', 'Two']
bin_n = 1/8
freqs = np.arange(0.5, 40.5, 0.5)  # Frequency range
n_cycles = freqs / 2.0  # Number of cycles for each frequency
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

test_size_n = 0.40  # 40% test size

threshold = 0.9 # variance threshold for feature selection

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
        epochs = mne.read_epochs(path, preload=True).resample(512)
        epochs = epochs.crop(tmin=0, tmax=2.0)

        # region epochs 
        epoch_df = epochs.to_data_frame()
        epoch_df['bin'] = (epoch_df['time'] // bin_n).astype(int)

        # Slopes calculation
        epochs_slopes_df = epoch_df.copy().drop(columns=['Stim', 'condition'])

        epochs_slopes_df = (epochs_slopes_df
            .groupby(['epoch', 'bin'])
            .apply(lambda x: x[ch_names_list].apply(lambda y: np.polyfit(x['time'], y, 1)[0]), 
                include_groups=False)
            .reset_index()
        )

        epochs_slopes_df_long = epochs_slopes_df.melt(
            id_vars = ['epoch', 'bin'],
            value_vars = ch_names_list,
            var_name = 'channel',
            value_name = 'slope'
        )

        epochs_slopes_df_wide = (
            epochs_slopes_df_long
            .pivot(
                index=['epoch'],
                columns=['channel', 'bin'],
                values='slope'
            )
            .reset_index()
        )

        epochs_slopes_df_wide.columns = [
            f'slope_{ch}_bin_{b}' for ch, b in epochs_slopes_df_wide.columns]

        epochs_slopes_df_wide = epochs_slopes_df_wide.rename(
            columns={'slope_epoch_bin_': 'epoch'})

        
        # Binned mean amplitude calculation

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

        print(f'epochs data frame for {id} {cond} created')

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
        
        print(f'PSD data frame for {id} {cond} created')

        
        merged_df = (
        epochs_slopes_df_wide
        .merge(epoch_wide, on='epoch')
        .merge(power_df_binned_wide, on='epoch')
        )

        # Deleting unused variables to free up memory
        del epochs 
        del epoch_df, epoch_long, epoch_wide
        del power, power_df, power_df_avg, power_df_avg_long, power_df_avg_wide
        del power_df_binned, power_df_binned_wide
        del epochs_slopes_df, epochs_slopes_df_long, epochs_slopes_df_wide


        corr_matrix = merged_df.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        df_reduced = merged_df.drop(columns=to_drop)
        
        print(f"Removed {len(to_drop)} highly correlated columns from {id} {cond}.")

        df_reduced['condition'] = cond
        df_reduced['participant_id'] = id
        df_reduced = df_reduced.dropna(axis = 1, how='any')

        per_participant.append(df_reduced)

    final_df = pd.concat(per_participant, ignore_index=True)

    df = final_df.drop(columns=['participant_id'])
    df = df.dropna(axis = 1, how='any')

    print(f'Final data frame for {id} created:')
    print(df.head())

    # region feature selection and LSTM preparation

    # 1) Features/target
    X = df.drop(columns=['condition'])
    y = df['condition'].map({'One': 0, 'Two': 1}).astype(int)

    # 2) Train/val/test split (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\n=== Data Split for {id} ===")
    print(f"Train samples: {len(y_train)} (One: {sum(y_train==0)}, Two: {sum(y_train==1)})")
    print(f"Val samples: {len(y_val)} (One: {sum(y_val==0)}, Two: {sum(y_val==1)})")
    print(f"Test samples: {len(y_test)} (One: {sum(y_test==0)}, Two: {sum(y_test==1)})")


    # => 70% train, 15% val, 15% test

    # 3) Univariate feature selection
    univariant_selector = SelectKBest(score_func=f_classif, k=min(300, X_train.shape[1]))
    X_train_uni = univariant_selector.fit_transform(X_train, y_train)

    # Boolean mask of selected features
    selected_mask = univariant_selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"[Univariate] Selected top {len(selected_features)} features based on ANOVA F-test.")

    # Reduce all splits to selected features
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    # 5) Reshape for LSTM: (samples, timesteps, features)
    # We'll create sequences by grouping features into bins

    # First, filter to only features with 'bin_' in name
    selected_features = [f for f in selected_features if 'bin_' in f]

    # Determine number of timesteps based on bins in feature names
    bin_numbers = []
    for feat in selected_features:
        if 'bin_' in feat:
            bin_num = int(feat.split('bin_')[-1])
            bin_numbers.append(bin_num)

    n_timesteps = max(bin_numbers) + 1 if bin_numbers else 16  # Default to 16 if no bins found

    # Organize features by bin to ensure temporal ordering
    features_by_bin = {i: [] for i in range(n_timesteps)}
    for feat in selected_features:
        if 'bin_' in feat:
            bin_num = int(feat.split('bin_')[-1])
            if bin_num < n_timesteps:
                features_by_bin[bin_num].append(feat)

    # Ensure equal features per bin (use minimum to avoid mismatches)
    min_features_per_bin = min(len(features_by_bin[i]) for i in range(n_timesteps) if len(features_by_bin[i]) > 0)

    # Flatten to get temporally-ordered feature list
    ordered_features = []
    for i in range(n_timesteps):
        ordered_features.extend(features_by_bin[i][:min_features_per_bin])

    features_per_timestep = min_features_per_bin

    print(f"Total features after selection: {len(selected_features)}")
    print(f"n_timesteps: {n_timesteps}, features_per_timestep: {features_per_timestep}")
    print(f"Features per bin: {[len(features_by_bin[i]) for i in range(n_timesteps)]}")

    # Reorder datasets to match temporal structure
    X_train_ordered = X_train_sel[ordered_features]
    X_val_ordered = X_val_sel[ordered_features]
    X_test_ordered = X_test_sel[ordered_features]

    # Scale the reordered features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_ordered)
    X_val_scaled = scaler.transform(X_val_ordered)
    X_test_scaled = scaler.transform(X_test_ordered)

    # Reshape data
    X_train_reshaped = X_train_scaled.reshape(
        X_train_scaled.shape[0], n_timesteps, features_per_timestep
    )
    X_val_reshaped = X_val_scaled.reshape(
        X_val_scaled.shape[0], n_timesteps, features_per_timestep
    )
    X_test_reshaped = X_test_scaled.reshape(
        X_test_scaled.shape[0], n_timesteps, features_per_timestep
    )

    
    print(f"Reshaped data: {X_train_reshaped.shape} (samples, timesteps, features)")

    # 6) Class imbalance handling
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    print(f"Class weights: {class_weights}")

    def create_lstm_model(units=64, dropout_rate=0.3, lr=1e-3):
        model = Sequential()
        model.add(Input(shape=(n_timesteps, features_per_timestep)))
        model.add(LSTM(units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Combine train + val for CV
    X_train_cv = np.concatenate([X_train_reshaped, X_val_reshaped], axis=0)
    y_train_cv = np.concatenate([y_train, y_val], axis=0)


    # Clear previous models from memory
    tf.keras.backend.clear_session()

    keras_clf = KerasClassifier(
        model=create_lstm_model,   # SciKeras uses `model=`, not `build_fn`
        epochs=20,
        batch_size=32,
        verbose=0
    )

    # Hyperparameter grid
    param_grid = {
        # kwargs for create_lstm_model(...):
        'model__units': [32, 64, 128],
        'model__dropout_rate': [0.2, 0.4],
        'model__lr': [1e-3, 1e-4],

        # kwargs for the KerasClassifier itself:
        'epochs': [20, 40],
        'batch_size': [32, 64],
    }


    grid = GridSearchCV(
    estimator=keras_clf,
    param_grid=param_grid,
    cv=3,                 # 3-fold CV on train+val
    scoring='accuracy',
    n_jobs=-4              # keep 1 to avoid TF + multiprocessing headaches
    )

    # Fit with class weights for imbalance
    grid_result = grid.fit(
        X_train_cv,
        y_train_cv,
        class_weight=class_weights
    )

    print("Best hyperparameters:", grid_result.best_params_)
    print(f"Best CV accuracy: {grid_result.best_score_:.3f}")

    # Get best model and test it
    best_estimator = grid_result.best_estimator_
    best_model = best_estimator.model_ 

    test_loss, test_acc = best_model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"Test accuracy with best hyperparameters: {test_acc:.3f}")

    # Extracting model's performance metrics
    # --- 1. Predict on test set with best model ---
    proba_test = best_model.predict(X_test_reshaped).ravel()
    pred_test = (proba_test > 0.5).astype(int)

    # --- 2. Confusion matrix metrics ---
    cm = confusion_matrix(y_test, pred_test)
    tn, fp, fn, tp = cm.ravel()
    total_One = tn + fp
    total_Two = fn + tp

    acc_one = tn / total_One if total_One > 0 else 0.0
    acc_two = tp / total_Two if total_Two > 0 else 0.0

    test_auc = roc_auc_score(y_test, proba_test)
    overall_accuracy = (tn + tp) / (total_One + total_Two)

    # --- 3. Extract best hyperparameters from GridSearchCV ---
    best_params = grid_result.best_params_

    lstm_units     = best_params.get("model__units", None)
    dropout_rate   = best_params.get("model__dropout_rate", None)
    learning_rate  = best_params.get("model__lr", None)
    batch_size     = best_params.get("batch_size", None)
    epochs_trained = best_params.get("epochs", None)

    # optional / placeholder fields if not used:
    bidirectional = False
    l1_reg = 0.0
    l2_reg = 0.0

    # --- 4. Store results for the participant ---
    performance_row = pd.DataFrame({
        "participant_id": [id],
        "total_One": [total_One],
        "total_Two": [total_Two],
        "correct_One": [tn],
        "correct_Two": [tp],
        "accuracy_One": [acc_one],
        "accuracy_Two": [acc_two],
        "overall_accuracy": [overall_accuracy],
        "test_auc": [test_auc],
        
        # hyperparameters
        "lstm_units": [lstm_units],
        "dropout_rate": [dropout_rate],
        "learning_rate": [learning_rate],
        "batch_size": [batch_size],
        "bidirectional": [bidirectional],
        "epochs_trained": [epochs_trained],
        "l1_reg": [l1_reg],
        "l2_reg": [l2_reg]
    })
    
    summary_df = pd.concat([summary_df, performance_row], ignore_index=True)

# 1: 0-21
# 2: 22:39
summary_df.to_csv('CNV_LSTM_3_2.csv', index=False)

# --- 5. Print summary ---
print(f"\nParticipant {id} classification summary:")
print(f"Units: {lstm_units}, Dropout: {dropout_rate}, LR: {learning_rate}")
print(f"Batch Size: {batch_size}, Bidirectional: {bidirectional}")
print(f"Accuracy One: {acc_one:.3f}, Accuracy Two: {acc_two:.3f}")
print(f"Overall Accuracy: {overall_accuracy:.3f}, Test AUC: {test_auc:.3f}")
print(f"Epochs trained: {epochs_trained}")
print("-" * 80)


