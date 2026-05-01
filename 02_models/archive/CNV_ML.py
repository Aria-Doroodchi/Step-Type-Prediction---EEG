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

# Importing epochs data
for id in participant_ids:
    path_one_all = os.path.join('bad_interpolated/Epochs/CNV', f'{id}_CNV_One-epo.fif')
    path_two_all = os.path.join('bad_interpolated/Epochs/CNV', f'{id}_CNV_Two-epo.fif')
    globals()[f"{id}_One_epochs"] = mne.read_epochs(path_one_all)
    globals()[f"{id}_Two_epochs"] = mne.read_epochs(path_two_all)

P01_One_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_One-epo.fif')
P01_One_epochs = P01_One_epochs.crop(tmin=0, tmax=2)

P01_Two_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_Two-epo.fif')
P01_Two_epochs = P01_Two_epochs.crop(tmin=0, tmax=2)

ch_names_list = P01_One_epochs.ch_names.copy()
ch_names_list.remove('Stim')

P01_One_epochs_df = P01_One_epochs.to_data_frame()
P01_Two_epochs_df = P01_Two_epochs.to_data_frame()


P01_One_epochs_df['bin'] = (P01_One_epochs_df['time'] // bin_n).astype(int)
P01_Two_epochs_df['bin'] = (P01_Two_epochs_df['time'] // bin_n).astype(int)

P01_One_epochs_df = P01_One_epochs_df.drop(columns=['Stim', 'condition', 'time'])
P01_Two_epochs_df = P01_Two_epochs_df.drop(columns=['Stim', 'condition', 'time'])

P01_One_epochs_df = (P01_One_epochs_df
    .groupby(['epoch', 'bin'])
    .mean(numeric_only=True)
    .reset_index()
)
P01_Two_epochs_df = (P01_Two_epochs_df
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
P01_Two_epochs_df_long = P01_Two_epochs_df.melt(
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
P01_Two_epochs_df_wide = (
    P01_Two_epochs_df_long
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

P01_Two_epochs_df_wide.columns = [f'{ch}_bin_{b}' for ch, b in P01_Two_epochs_df_wide.columns]
P01_Two_epochs_df_wide = P01_Two_epochs_df_wide.rename(
    columns={'epoch_bin_': 'epoch'})

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
power_Two = P01_Two_epochs.compute_tfr(
    method='morlet',
    freqs=freqs,
    n_cycles=n_cycles,
    return_itc=False,
    average=False  # Keep individual epochs
)


power_One_df = power_One.to_data_frame()
power_Two_df = power_Two.to_data_frame()

# Define frequency bands
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

freq_bands_One = [
    power_One_df['freq'].between(0.5, 4.0),  # Delta
    power_One_df['freq'].between(4.0, 8.0),  # Theta
    power_One_df['freq'].between(8.0, 13.0),  # Alpha
    power_One_df['freq'].between(13.0, 30.0),  # Beta
    power_One_df['freq'].between(30.0, 40.0)   # Gamma
]

freq_bands_Two = [
    power_Two_df['freq'].between(0.5, 4.0),  # Delta
    power_Two_df['freq'].between(4.0, 8.0),  # Theta
    power_Two_df['freq'].between(8.0, 13.0),  # Alpha
    power_Two_df['freq'].between(13.0, 30.0),  # Beta
    power_Two_df['freq'].between(30.0, 40.0)   # Gamma
]


power_One_df['freq'] = np.select(freq_bands_One, freq_band_names, default=power_One_df['freq'])
power_Two_df['freq'] = np.select(freq_bands_Two, freq_band_names, default=power_Two_df['freq'])


power_One_df['bin'] = (power_One_df['time'] // bin_n).astype(int)
power_One_df = power_One_df.drop(columns=['time', 'condition'])

power_Two_df['bin'] = (power_Two_df['time'] // bin_n).astype(int)
power_Two_df = power_Two_df.drop(columns=['time', 'condition'])

power_One_df_avg = (power_One_df
    .groupby(['freq', 'epoch', 'bin'])
    .mean(numeric_only=True)
    .reset_index()
)
power_Two_df_avg = (power_Two_df
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
power_Two_df_long = power_Two_df_avg.melt(
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

power_df_Two_wide = (
    power_Two_df_long
    .pivot(index=['epoch', 'bin'],
              columns=['channel', 'freq'], 
              values='power')
    .reset_index()
)
power_df_Two_wide.columns = [f'{ch}_{band}' for ch, band in power_df_Two_wide.columns]


power_df_One_wide = power_df_One_wide.rename(
    columns={'epoch_': 'epoch',
              'bin_': 'bin'})
power_df_Two_wide = power_df_Two_wide.rename(
    columns={'epoch_': 'epoch',
              'bin_': 'bin'})


power_df_binned_One = power_df_One_wide.melt(
    id_vars = ['epoch', 'bin'],
    value_vars = power_df_One_wide.columns[2:],
    var_name = 'channel_freqband',
    value_name = 'power'
)
power_df_binned_Two = power_df_Two_wide.melt(
    id_vars = ['epoch', 'bin'],
    value_vars = power_df_Two_wide.columns[2:],
    var_name = 'channel_freqband',
    value_name = 'power'
)

# this is actually wide, not long
power_df_One_binned_long = (
    power_df_binned_One
    .pivot(
        index=['epoch'],
        columns=['channel_freqband', 'bin'],
        values='power')
    .reset_index()
)
power_df_Two_binned_long = (
    power_df_binned_Two
    .pivot(
        index=['epoch'],
        columns=['channel_freqband', 'bin'],
        values='power')
    .reset_index()
)

power_df_One_binned_long.columns = [f'{chfb}_bin_{b}' for chfb, b in power_df_One_binned_long.columns]
power_df_Two_binned_long.columns = [f'{chfb}_bin_{b}' for chfb, b in power_df_Two_binned_long.columns]

power_df_binned_One = power_df_One_binned_long.rename(
    columns={'epoch_bin_': 'epoch'})
power_df_binned_Two = power_df_Two_binned_long.rename(
    columns={'epoch_bin_': 'epoch'})



One_merged = pd.merge(
    P01_One_epochs_df_wide,
    power_df_binned_One,
    on=['epoch'])

One_merged['condition'] = 'One'

Two_merged = pd.merge(
    P01_Two_epochs_df_wide,
    power_df_binned_Two,
    on=['epoch'])
Two_merged['condition'] = 'Two'

combined_df = pd.concat([One_merged, Two_merged], ignore_index=True)
combined_df.drop(columns=['epoch'], inplace=True)

df = combined_df.copy()

# endregion PSD
###############################################################################
# region ML 

# Core ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC


# ------------------------------------------------------------
# Assume your dataframe is called df and includes 'condition'
# with string values "One" and "Two", and all other columns are numeric.
# ------------------------------------------------------------

# 1) Split features/target and encode labels
X = df.drop(columns=['condition'])
y = df['condition'].map({'One': 0, 'Two': 1}).astype(int)

# 2) Train/val/test split (stratified to keep class balance)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
# Now: 70% train, 15% val, 15% test

# 3) Scale features (fit on train only)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# 4) Handle class imbalance (optional but recommended)
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
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

summary = pd.DataFrame({
    "Condition": ["One", "Two"],
    "Correct":   [correct_one, correct_two],
    "Total":     [total_one, total_two],
    "Accuracy":  [acc_one, acc_two]
})

summary

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred_test))

print("\nClassification report:")
print(classification_report(y_test, pred_test, target_names=['One','Two']))

print("ROC AUC:", roc_auc_score(y_test, proba_test))

# ------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay

from xgboost import XGBClassifier


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

# 3) Class imbalance handling (scale_pos_weight = neg/pos, where pos = label 1)
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

# 4) Build XGBoost model
xgb = XGBClassifier(
    n_estimators=2000,           # large cap; early stopping will cut it
    learning_rate=0.02,          # smaller LR pairs well with early stopping
    max_depth=6,                 # start reasonable; tune later
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,              # L2
    reg_alpha=0.0,               # L1
    gamma=0.0,                   # split gain threshold
    scale_pos_weight=scale_pos_weight,
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=100,
    tree_method='hist',          # fast, memory-efficient; use 'gpu_hist' if you have supported GPU
    random_state=42,
    n_jobs=22
)

# 5) Train with early stopping on validation
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# 6) Evaluate on test
proba_test = xgb.predict_proba(X_test)[:, 1]
pred_test  = (proba_test >= 0.5).astype(int)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred_test))

print("\nClassification report:")
print(classification_report(y_test, pred_test, target_names=['One','Two']))

print("ROC AUC:", roc_auc_score(y_test, proba_test))

# 7) Quick ROC curve
RocCurveDisplay.from_predictions(y_test, proba_test)
plt.title("XGBoost ROC (test)")
plt.show()

# 8) Top feature importances (gain)
importances = pd.Series(xgb.get_booster().get_score(importance_type='gain'))
# Align to dataframe columns; missing keys -> 0
importances = importances.reindex(X.columns, fill_value=0.0).sort_values(ascending=False)

print("\nTop 15 features by gain:")
print(importances.head(15))



cm = confusion_matrix(y_test, pred_test)
tn, fp, fn, tp = cm.ravel()

# Class 0 (One)
acc_one = tn / (tn + fp)

# Class 1 (Two)
acc_two = tp / (tp + fn)

print("Accuracy for class One (0):", acc_one)
print("Accuracy for class Two (1):", acc_two)

total_one = tn + fp
total_two = fn + tp

total_accuracy = (tn + tp) / (total_one + total_two)
