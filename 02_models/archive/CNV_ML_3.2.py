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

import warnings

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from xgboost import XGBClassifier



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
                    'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21', 'P23', 'P24',
                    'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']



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
        epochs = mne.read_epochs(path, preload=True)
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
    
   # 1) Features/target
    X = df.drop(columns=['condition'])
    y = df['condition'].map({'One': 0, 'Two': 1}).astype(int)


    # 2) Train/val/test split (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    # => 70% train, 15% val, 15% test

    univariant_selector = SelectKBest(score_func=f_classif, k=500)
    X = univariant_selector.fit_transform(X, y)

    # Boolean mask of selected features
    selected_mask = univariant_selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"[Univariate] Selected top {len(selected_features)} features based on ANOVA F-test.")

    # Reduce all splits to selected features, still as DataFrames
    X_train = X_train[selected_features]
    X_val   = X_val[selected_features]
    X_test  = X_test[selected_features]


    
    # 3) Class imbalance handling (scale_pos_weight = neg/pos, where pos = label 1)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    rfecv_base = XGBClassifier(
    n_estimators=600,          # smaller to keep CV fast
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.1,                 # light split-threshold regularization
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    random_state=42,
    n_jobs=22
    )

    # splots the dataset into 5 folds for cross-validation
    cv_n = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    # 4) RFECV for feature selection
    selector = RFECV(
    estimator=rfecv_base,
    step=0.2,                         # remove ~20% lowest-ranked each iteration
    cv=cv_n,
    scoring='roc_auc',
    n_jobs=22,
    min_features_to_select=50         # safeguard; adjust to your dataset size
    )

    selector.fit(X_train, y_train)

    # Get masks / names of selected features
    selected_mask = selector.support_
    selected_features = X_train.columns[selected_mask].tolist()
    print(f"[RFECV] Selected {len(selected_features)} features out of {X_train.shape[1]}.")

    # Transform splits to selected features
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    # Optional: monitor CV performance from RFECV
    print(f"[RFECV] Best CV score: {selector.cv_results_['mean_test_score'].max():.3f}")

    # -----------------------------
    # B) Final XGB on selected features with early stopping
    # -----------------------------
    xgb_final = XGBClassifier(
    n_estimators=2000,         # large cap; early stopping will cut it
    learning_rate=0.03,        # small LR pairs well with early stopping
    max_depth=4,               # you can later tune 4–6
    min_child_weight=3.0,      # a touch more conservative after selection
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=3.0,            # slightly stronger L2
    reg_alpha=0.5,             # L1 for sparsity in splits
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=100,
    tree_method='hist',        # use 'gpu_hist' if supported
    random_state=42,
    n_jobs=22
    )

    xgb_final.fit(
    X_train_sel, y_train,
    eval_set=[(X_val_sel, y_val)],
    verbose=False
    )

    # 6) Evaluate on test (selected features)
    proba_test = xgb_final.predict_proba(X_test_sel)[:, 1]
    pred_test  = (proba_test >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred_test)
    tn, fp, fn, tp = cm.ravel()

    total_One = tn + fp
    total_Two = fn + tp

    acc_one = tn / total_One if total_One > 0 else 0.0
    acc_two = tp / total_Two if total_Two > 0 else 0.0
    auc     = roc_auc_score(y_test, proba_test)

    temp_df = pd.DataFrame({
        'total_One': [total_One],
        'total_Two': [total_Two],
        'correct_One': [tn],
        'correct_Two': [tp]
    })

    summary_df = pd.concat([summary_df, temp_df], ignore_index=True)

    print(f'Participant {id} classification report:')
    print(f'Accuracy One: {acc_one:.3f}'
          f', Accuracy Two: {acc_two:.3f}'
          f', Overall Accuracy: {(tn + tp) / (total_One + total_Two):.3f}\n')


total_One_final = summary_df['total_One'].sum()
total_Two_final = summary_df['total_Two'].sum()
correct_One_final = summary_df['correct_One'].sum()
correct_Two_final = summary_df['correct_Two'].sum()

overall_acc_one = correct_One_final / total_One_final if total_One_final > 0 else 0.0
overall_acc_two = correct_Two_final / total_Two_final if total_Two_final > 0 else 0.0
overall_accuracy = (correct_One_final + correct_Two_final) / (total_One_final + total_Two_final) if (total_One_final + total_Two_final) > 0 else 0.0

print(f'Overall Accuracy One: {overall_acc_one:.3f}')
#59.6%
print(f'Overall Accuracy Two: {overall_acc_two:.3f}')
#54.4%
print(f'Overall Accuracy: {overall_accuracy:.3f}')
#57.1%

