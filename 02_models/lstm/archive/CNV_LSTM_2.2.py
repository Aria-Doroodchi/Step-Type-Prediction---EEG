import os
import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from mne.minimum_norm import apply_inverse, make_inverse_operator
import pprint
from collections import Counter
import concurrent.futures
import gc
import psutil
import time
import warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')
from contextlib import contextmanager
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif

# ============================================================================
# region Memory Monitoring Setup
# ============================================================================

def get_mem_gb() -> float:
    """Return current process RSS in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def log_memory(tag: str):
    """Print current RAM usage with a label."""
    print(f"[MEM] {tag}: {get_mem_gb():.3f} GB")


@contextmanager
def mem_section(name: str):
    """Context manager to measure memory + time for a code block."""
    mem_before = get_mem_gb()
    t0 = time.time()
    print(f"\n[MEM-SECTION START] {name} | {mem_before:.3f} GB")
    try:
        yield
    finally:
        mem_after = get_mem_gb()
        t1 = time.time()
        print(f"[MEM-SECTION END]   {name} | {mem_after:.3f} GB "
              f"(Δ={mem_after - mem_before:+.3f} GB, {t1 - t0:.1f}s)")

# endregion Memory Monitoring Setup

# ============================================================================
# region Setup
# ============================================================================

log_memory("Script initialization")

new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)
print(f"Updated working directory: {os.getcwd()}")

log_memory("After directory change")

# endregion setup 

# region global parameters

participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
                  'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21', 'P23', 'P24',
                  'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']

# LSTM hyperparameters
lstm_units_list = [32, 128]
dropout_rate_list = [0.2, 0.5]
learning_rate_list = [0.001, 0.0001]
batch_size_list = [16, 64]
bidirectional_list = [True, False]
l1_list = [0.0, 0.01]
l2_list = [0.0, 0.01]

conditions = ['One', 'Two']
bin_n = 1/8
freqs = np.arange(0.5, 40.5, 0.5)
n_cycles = freqs / 2.0
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

test_size_n = 0.40
threshold = 0.9

# Pruning configuration
PRUNING_INTERVAL = 5   # Prune very frequently
TOP_N_TO_KEEP = 1      # Keep only top 15 results

log_memory("After global parameters setup")

# epoch info for data wrangling
with mem_section("Loading P01 epoch info"):
    P01_One_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_One-epo.fif')
    ch_names_list = P01_One_epochs.ch_names.copy()
    ch_names_list.remove('Stim')

log_memory("After P01 epoch info loaded")

# endregion global parameters

# ============================================================================
# region pruning function 
# ============================================================================

def prune_results(rows_list, top_n=TOP_N_TO_KEEP):
    """
    Combine all rows, sort by overall_accuracy (descending), 
    and keep only the top N results.
    """
    log_memory(f"Before pruning - {len(rows_list)} rows")
    
    if not rows_list:
        return []
    
    combined_df = pd.concat(rows_list, ignore_index=True)
    # Sort by overall_accuracy descending, then by test_auc descending
    pruned_df = combined_df.sort_values(
        by=['overall_accuracy', 'test_auc'], 
        ascending=[False, False]
    ).head(top_n)
    
    print(f"\n[PRUNING] Reduced from {len(combined_df)} to {len(pruned_df)} rows")
    print(f"[PRUNING] Top accuracy: {pruned_df['overall_accuracy'].iloc[0]:.3f}")
    print(f"[PRUNING] Lowest kept accuracy: {pruned_df['overall_accuracy'].iloc[-1]:.3f}\n")
    
    # Delete the old combined dataframe
    del combined_df
    gc.collect()  # Force garbage collection
    
    log_memory(f"After pruning - {len(pruned_df)} rows kept")
    
    # Return as list of single-row DataFrames for consistency
    return [pruned_df.iloc[[i]] for i in range(len(pruned_df))]

# endregion pruning function

# ============================================================================
# Region single participant function
# ============================================================================

def run_single_participant(pid: str) -> pd.DataFrame:
    """
    Run the full pipeline (feature extraction + hyperparameter search + evaluation)
    for a single participant and return a DataFrame with all rows for that participant.
    """
    print(f"\n########################")
    print(f"Starting participant {pid}")
    print(f"########################\n")

    log_memory(f"{pid} - start")

    per_participant = []

    # region feature extraction

    for cond in conditions:
        with mem_section(f"{pid} {cond} - load & preprocess epochs"):
            path = os.path.join('bad_interpolated/Epochs/CNV', f'{pid}_CNV_{cond}-epo.fif')
            epochs = mne.read_epochs(path, preload=True).resample(128)
            epochs = epochs.crop(tmin=0, tmax=2.0)

            epoch_df = epochs.to_data_frame()
            epoch_df['bin'] = (epoch_df['time'] // bin_n).astype(int)

        log_memory(f"{pid} {cond} - after epoch loading")

        # Slopes calculation
        with mem_section(f"{pid} {cond} - slopes DF"):
            epochs_slopes_df = epoch_df.copy().drop(columns=['Stim', 'condition'])

            epochs_slopes_df = (
                epochs_slopes_df
                .groupby(['epoch', 'bin'])
                .apply(lambda x: x[ch_names_list].apply(lambda y: np.polyfit(x['time'], y, 1)[0]),
                    include_groups=False)
                .reset_index()
            )

            epochs_slopes_df_long = epochs_slopes_df.melt(
                id_vars=['epoch', 'bin'],
                value_vars=ch_names_list,
                var_name='channel',
                value_name='slope'
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
                f'slope_{ch}_bin_{b}' for ch, b in epochs_slopes_df_wide.columns
            ]
            epochs_slopes_df_wide = epochs_slopes_df_wide.rename(
                columns={'slope_epoch_bin_': 'epoch'}
            )

        log_memory(f"{pid} {cond} - after slopes calculation")

        # Binned mean amplitude
        with mem_section(f"{pid} {cond} - amplitude DF"):

            epoch_df_amp = epoch_df.drop(columns=['Stim', 'condition', 'time'])

            epoch_df_amp = (
                epoch_df_amp
                .groupby(['epoch', 'bin'])
                .mean(numeric_only=True)
                .reset_index()
            )

            epoch_long = epoch_df_amp.melt(
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

        log_memory(f"{pid} {cond} - after amplitude calculation")

        # PSD
        with mem_section(f"{pid} {cond} - TFR/PSD"):
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

            power_df_avg = (
                power_df
                .groupby(['freq', 'epoch', 'bin'])
                .mean(numeric_only=True)
                .reset_index()
            )

            power_df_avg_long = power_df_avg.melt(
                id_vars=['freq', 'epoch', 'bin'],
                value_vars=ch_names_list,
                var_name='channel',
                value_name='power'
            )

            power_df_avg_wide = (
                power_df_avg_long
                .pivot(
                    index=['epoch', 'bin'],
                    columns=['channel', 'freq'],
                    values='power'
                )
                .reset_index()
            )

            power_df_avg_wide.columns = [
                f'{ch}_{band}' for ch, band in power_df_avg_wide.columns
            ]

            power_df_avg_wide = power_df_avg_wide.rename(
                columns={'epoch_': 'epoch', 'bin_': 'bin'}
            )

            power_df_binned = power_df_avg_wide.melt(
                id_vars=['epoch', 'bin'],
                value_vars=power_df_avg_wide.columns[2:],
                var_name='channel_freqband',
                value_name='power'
            )

            power_df_binned_wide = (
                power_df_binned
                .pivot(
                    index=['epoch'],
                    columns=['channel_freqband', 'bin'],
                    values='power'
                )
                .reset_index()
            )

            power_df_binned_wide.columns = [
                f'{chfb}_bin_{b}' for chfb, b in power_df_binned_wide.columns
            ]

            power_df_binned_wide = power_df_binned_wide.rename(
                columns={'epoch_bin_': 'epoch'}
            )

        log_memory(f"{pid} {cond} - after PSD calculation")

        with mem_section(f"{pid} {cond} - merge features"):

            merged_df = (
                epochs_slopes_df_wide
                .merge(epoch_wide, on='epoch')
                .merge(power_df_binned_wide, on='epoch')
            )

        log_memory(f"{pid} {cond} - after feature merge")

        # endregion feature extraction

        # region feature selection 
        with mem_section(f"{pid} {cond} - corr + drop correlated"):

            corr_matrix = merged_df.select_dtypes(include=[np.number]).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            df_reduced = merged_df.drop(columns=to_drop)

            df_reduced['condition'] = cond
            df_reduced['participant_id'] = pid
            df_reduced = df_reduced.dropna(axis=1, how='any')

        log_memory(f"{pid} {cond} - after correlation filtering")

        per_participant.append(df_reduced)

    with mem_section(f"{pid} - concat conditions"):
        final_df = pd.concat(per_participant, ignore_index=True)

        df = final_df.drop(columns=['participant_id'])
        df = df.dropna(axis=1, how='any')

        del epoch_df_amp, epoch_long, epoch_wide
        del epochs_slopes_df, epochs_slopes_df_long, epochs_slopes_df_wide
        del power_df, power_df_avg, power_df_avg_long, power_df_avg_wide, power_df_binned_wide
        gc.collect()

    log_memory(f"{pid} - after concat and cleanup")

    with mem_section(f"{pid} - train/val/test split + SelectKBest"):

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

        # 3) Univariate feature selection
        univariant_selector = SelectKBest(score_func=f_classif, k=min(300, X_train.shape[1]))
        X_train_uni = univariant_selector.fit_transform(X_train, y_train)

        selected_mask = univariant_selector.get_support()
        selected_features = X_train.columns[selected_mask].tolist()

    log_memory(f"{pid} - after train/test split")

    # Reduce all splits to selected features
    X_train_sel = X_train[selected_features]
    X_val_sel   = X_val[selected_features]
    X_test_sel  = X_test[selected_features]

    # 5) Reshape for LSTM
    selected_features = [f for f in selected_features if 'bin_' in f]

    bin_numbers = []
    for feat in selected_features:
        if 'bin_' in feat:
            bin_num = int(feat.split('bin_')[-1])
            bin_numbers.append(bin_num)

    n_timesteps = max(bin_numbers) + 1 if bin_numbers else 16

    features_by_bin = {i: [] for i in range(n_timesteps)}
    for feat in selected_features:
        if 'bin_' in feat:
            bin_num = int(feat.split('bin_')[-1])
            if bin_num < n_timesteps:
                features_by_bin[bin_num].append(feat)

    min_features_per_bin = min(
        len(features_by_bin[i]) for i in range(n_timesteps) if len(features_by_bin[i]) > 0
    )

    ordered_features = []
    for i in range(n_timesteps):
        ordered_features.extend(features_by_bin[i][:min_features_per_bin])

    features_per_timestep = min_features_per_bin

    X_train_ordered = X_train_sel[ordered_features]
    X_val_ordered = X_val_sel[ordered_features]
    X_test_ordered = X_test_sel[ordered_features]

    with mem_section(f"{pid} - scaling and reshaping"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_ordered)
        X_val_scaled = scaler.transform(X_val_ordered)
        X_test_scaled = scaler.transform(X_test_ordered)

        X_train_reshaped = X_train_scaled.reshape(
            X_train_scaled.shape[0], n_timesteps, features_per_timestep
        )
        X_val_reshaped = X_val_scaled.reshape(
            X_val_scaled.shape[0], n_timesteps, features_per_timestep
        )
        X_test_reshaped = X_test_scaled.reshape(
            X_test_scaled.shape[0], n_timesteps, features_per_timestep
        )

    log_memory(f"{pid} - after scaling and reshaping")

    # 6) Class imbalance
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

    # endregion feature selection

    # region hyperparameter tuning

    rows = []  # collect all rows for this participant
    iteration_count = 0

    log_memory(f"{pid} - starting hyperparameter tuning")

    for lstm_units in lstm_units_list:
        for dropout_rate in dropout_rate_list:
            for learning_rate in learning_rate_list:
                for batch_size in batch_size_list:
                    for use_bidirectional in bidirectional_list:
                        for l1_reg in l1_list:
                            for l2_reg in l2_list:
                                    
                                with mem_section(
                                f"{pid} - model iter {iteration_count} "
                                f"(units={lstm_units}, dr={dropout_rate}, "
                                f"lr={learning_rate}, bs={batch_size}, "
                                f"bi={use_bidirectional}, l1={l1_reg}, l2={l2_reg})"
                                ):

                                    tf.keras.backend.clear_session()
                                    log_memory(f"{pid} - iter {iteration_count} - after clear_session")

                                    model = Sequential()
                                    model.add(Input(shape=(n_timesteps, features_per_timestep)))

                                    if use_bidirectional:
                                        model.add(Bidirectional(
                                            LSTM(
                                                lstm_units,
                                                return_sequences=True,
                                                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                                            )
                                        ))
                                    else:
                                        model.add(LSTM(
                                            lstm_units,
                                            return_sequences=True,
                                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                                        ))

                                    model.add(Dropout(dropout_rate))
                                    model.add(BatchNormalization())

                                    if use_bidirectional:
                                        model.add(Bidirectional(
                                            LSTM(
                                                lstm_units // 2,
                                                return_sequences=False,
                                                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
                                            )
                                        ))
                                    else:
                                        model.add(LSTM(
                                            lstm_units // 2,
                                            return_sequences=False,
                                            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
                                        ))

                                    model.add(Dropout(dropout_rate))
                                    model.add(BatchNormalization())

                                    model.add(Dense(32, activation='relu',
                                                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
                                    model.add(Dropout(dropout_rate))
                                    model.add(Dense(1, activation='sigmoid'))

                                    log_memory(f"{pid} - iter {iteration_count} - after model build")

                                    optimizer = Adam(learning_rate=learning_rate)
                                    model.compile(
                                        optimizer=optimizer,
                                        loss='binary_crossentropy',
                                        metrics=['accuracy', AUC(name='auc')]
                                    )

                                    log_memory(f"{pid} - iter {iteration_count} - after compile")

                                    early_stopping = EarlyStopping(
                                        monitor='val_auc',
                                        patience=20,
                                        restore_best_weights=True,
                                        mode='max'
                                    )

                                    reduce_lr = ReduceLROnPlateau(
                                        monitor='val_loss',
                                        factor=0.5,
                                        patience=10,
                                        min_lr=1e-7,
                                        verbose=0
                                    )

                                    history = model.fit(
                                        X_train_reshaped, y_train,
                                        validation_data=(X_val_reshaped, y_val),
                                        epochs=200,
                                        batch_size=batch_size,
                                        class_weight=class_weights,
                                        callbacks=[early_stopping, reduce_lr],
                                        verbose=0
                                    )

                                    log_memory(f"{pid} - iter {iteration_count} - after training")

                                    # Extract only the metrics we need before clearing history
                                    final_val_auc = history.history['val_auc'][-1]
                                    epochs_trained = len(history.history['loss'])
                                    
                                    # Clear history to free memory
                                    del history
                                    
                                    proba_test = model.predict(X_test_reshaped, verbose=0).flatten()
                                    pred_test = (proba_test >= 0.5).astype(int)

                                    log_memory(f"{pid} - iter {iteration_count} - after prediction")

                                    cm = confusion_matrix(y_test, pred_test)
                                    tn, fp, fn, tp = cm.ravel()

                                    total_One = tn + fp
                                    total_Two = fn + tp

                                    acc_one = tn / total_One if total_One > 0 else 0.0
                                    acc_two = tp / total_Two if total_Two > 0 else 0.0
                                    auc = roc_auc_score(y_test, proba_test)
                                    overall_accuracy = (
                                        (tn + tp) / (total_One + total_Two)
                                        if (total_One + total_Two) > 0 else 0.0
                                    )
                                    
                                    # Clean up predictions to free memory
                                    del proba_test, pred_test

                                    temp_df = pd.DataFrame({
                                        'participant_id': [pid],
                                        'total_One': [total_One],
                                        'total_Two': [total_Two],
                                        'correct_One': [tn],
                                        'correct_Two': [tp],
                                        'accuracy_One': [acc_one],
                                        'accuracy_Two': [acc_two],
                                        'overall_accuracy': [overall_accuracy],
                                        'test_auc': [auc],
                                        'val_auc': [final_val_auc],
                                        'lstm_units': [lstm_units],
                                        'dropout_rate': [dropout_rate],
                                        'learning_rate': [learning_rate],
                                        'batch_size': [batch_size],
                                        'bidirectional': [use_bidirectional],
                                        'epochs_trained': [epochs_trained],
                                        'l1_reg': [l1_reg],
                                        'l2_reg': [l2_reg]
                                    })

                                    rows.append(temp_df)
                                    
                                    # Delete model explicitly to free CPU memory
                                    del model
                                    tf.keras.backend.clear_session()
                                    gc.collect()
                                    
                                    log_memory(f"{pid} - iter {iteration_count} - after cleanup")
                                    
                                    iteration_count += 1

                                    # Force garbage collection every 10 iterations
                                    if iteration_count % 10 == 0:
                                        gc.collect()
                                        log_memory(f"{pid} - after GC at iteration {iteration_count}")

                                    # Pruning: every PRUNING_INTERVAL rows, keep only top results
                                    if iteration_count % PRUNING_INTERVAL == 0:
                                        rows = prune_results(rows, top_n=TOP_N_TO_KEEP)
                                        gc.collect()  # Force garbage collection after pruning
                                        log_memory(f"{pid} - after pruning at iteration {iteration_count}")

                                    print(f'Participant {pid} classification report:')
                                    print(f'Accuracy One: {acc_one:.3f}, Accuracy Two: {acc_two:.3f}')
                                    print(f'Overall Accuracy: {overall_accuracy:.3f}, Test AUC: {auc:.3f}')
                                    print('-' * 80)

    print("=" * 80)
    print("=" * 80)
    log_memory(f"{pid} - finished hyperparameter tuning")

    participant_summary_df = pd.concat(rows, ignore_index=True)
    
    # Final cleanup for this participant
    del rows
    gc.collect()
    
    log_memory(f"{pid} - final cleanup complete")
    
    return participant_summary_df

    # endregion hyperparameter tuning


# region main & parallel

if __name__ == "__main__":
        log_memory("main - start")

        max_workers = 2
        
        print(f"Running with {max_workers} parallel workers")
        print(f"Pruning every {PRUNING_INTERVAL} iterations, keeping top {TOP_N_TO_KEEP}")
        
        with mem_section("main - parallel execution"):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map over participants; each process returns a DataFrame
                dfs = list(executor.map(run_single_participant, participant_ids))
        
        log_memory("main - after executor.map")

        # Combine all participants' results into one summary_df (same format as before)
        with mem_section("main - concatenating results"):
            summary_df = pd.concat(dfs, ignore_index=True)
        
        # Cleanup intermediate results
        del dfs
        gc.collect()
        log_memory("main - after concat summary_df")

        # ------------------ Your original final aggregation code ------------------ #
        with mem_section("main - generating final report"):
            report_df = (
                summary_df
                .sort_values(['participant_id', 'overall_accuracy'], ascending=[True, False])
                .drop_duplicates(subset='participant_id', keep='first')
            )

            report_df.to_csv('CNV_LSTM_report.csv', index=False)

        log_memory("main - after report generation")

        total_One_final = report_df['total_One'].sum()
        total_Two_final = report_df['total_Two'].sum()
        correct_One_final = report_df['correct_One'].sum()
        correct_Two_final = report_df['correct_Two'].sum()

        overall_acc_one = correct_One_final / total_One_final if total_One_final > 0 else 0.0
        overall_acc_two = correct_Two_final / total_Two_final if total_Two_final > 0 else 0.0
        overall_accuracy = (
            (correct_One_final + correct_Two_final) /
            (total_One_final + total_Two_final) if (total_One_final + total_Two_final) > 0 else 0.0
        )

        print(f'\n=== FINAL RESULTS ===')
        print(f'Overall Accuracy One: {overall_acc_one:.3f}')
        print(f'Overall Accuracy Two: {overall_acc_two:.3f}')
        print(f'Overall Accuracy: {overall_accuracy:.3f}')
        print(f'\nMean Test AUC: {report_df["test_auc"].mean():.3f}')
        print(f'Mean Val AUC: {report_df["val_auc"].mean():.3f}')
        print(f'\nBidirectional usage:')
        print(report_df['bidirectional'].value_counts())
        print(f'\nMean epochs trained: {report_df["epochs_trained"].mean():.1f}')
        
        log_memory("main - script complete")

# endregion main & parallel