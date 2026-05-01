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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


new_directory = 'C:/Users/Aria/OneDrive - The University of Western Ontario/MSc/Thesis/Data/Participants'
os.chdir(new_directory)

# Get the updated working directory
updated_working_directory = os.getcwd()
print(f"Updated working directory: {updated_working_directory}")
# endregion setting up the environment

###############################################################################
# region variables and parameters

# data frame variables
participant_ids = ['P01', 'P02', 'P03', 'P05', 'P06', 'P07', 'P08', 'P10', 'P11',
                    'P12', 'P13', 'P14', 'P15', 'P16', 'P18', 'P19', 'P21', 'P23', 'P24',
                    'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P33', 'P35', 'P37', 'P39']




conditions = ['One', 'Two']

min_time_var = 0 #onset time for predictions. must be betweeen 0 and 2

bin_n = 1/8 # over a 2 second window 

freqs = np.arange(0.5, 40.5, 0.5)  # Frequency range
n_cycles = freqs / 2.0  # Number of cycles for each frequency
freq_band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# source localization parameters


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

LORETA_epoch_rows = []


# model parameters

max_depth_list = [4]
min_child_weight_list = [1]
reg_alpha_list = [0, 0.1]
gamma_list = [0, 0.1, 0.3, 0.5]
reg_lambda_list = [1, 3, 5, 10]
colsample_bytree_list = [0.6, 0.7]
colsample_bylevel_list = [0.6, 0.8,]
learning_rate_list = [0.01, 0.03, 0.05]
sub_sample_list = [0.6, 0.8, 1.0]


param_grid = {
    "max_depth": max_depth_list,
    "min_child_weight": min_child_weight_list,
    "reg_alpha": reg_alpha_list,
    "gamma": gamma_list,
    "reg_lambda": reg_lambda_list,
    "colsample_bytree": colsample_bytree_list,
    "colsample_bylevel": colsample_bylevel_list,
    "learning_rate": learning_rate_list,
    "subsample": sub_sample_list
}


test_size_n = 0.30  # 30% test size

threshold = 0.9 # variance threshold for feature selection

# endregion variables and parameters

# epoch info for data wrangling
P01_One_epochs = mne.read_epochs('bad_interpolated/Epochs/CNV/P01_CNV_One-epo.fif')
ch_names_list = P01_One_epochs.ch_names.copy()
ch_names_list.remove('Stim')

# data frame for storing model performance results
summary_df = pd.DataFrame()
feature_importance_df = pd.DataFrame()

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

        # endregion source localization

        epochs = epochs.crop(tmin= min_time_var, tmax=2.0)
 

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

        # PSD

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
        .merge(bm_df, on='epoch')
        )

        # Deleting unused variables to free up memory
        del epochs 
        del epoch_df, epoch_long, epoch_wide
        del power, power_df, power_df_avg, power_df_avg_long, power_df_avg_wide
        del power_df_binned, power_df_binned_wide
        del epochs_slopes_df, epochs_slopes_df_long, epochs_slopes_df_wide
        del bm_df

        # endregion data wrangling 

        # region feature selection 

        # Remove highly correlated features
        corr_matrix = merged_df.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        df_reduced = merged_df.drop(columns=to_drop)

        print(f"Removed {len(to_drop)} highly correlated columns from {id} {cond}.")

        df_reduced['condition'] = cond
        df_reduced['participant_id'] = id
        df_reduced = df_reduced.dropna(axis=1, how='any')

        per_participant.append(df_reduced)

    final_df = pd.concat(per_participant, ignore_index=True)

    df = final_df.drop(columns=['participant_id'])
    df = df.dropna(axis=1, how='any')

    print(f'Final data frame for {id} created:')
    print(df.head())

    # 1) Features/target
    X = df.drop(columns=['condition'])
    y = df['condition'].map({'One': 0, 'Two': 1}).astype(int)

    # 2) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_n, random_state=1, stratify=y
    )

    # univariant feature selection
    univariant_selector = SelectKBest(score_func=f_classif, k=500)
    X_train_selected = univariant_selector.fit_transform(X_train, y_train)

    # Boolean mask of selected features
    selected_mask = univariant_selector.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"[Univariate] Selected top {len(selected_features)} features based on ANOVA F-test.")

    # Reduce splits to selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # 3) Class imbalance handling (scale_pos_weight = neg/pos, where pos = label 1)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # RFECV feature selection

    # RFECV 
    rfecv_base = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        reg_alpha=0.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=1,
        n_jobs=16
    )

    # splits the dataset into folds for cross-validation
    cv_n = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # RFECV for feature selection
    selector = RFECV(
        estimator=rfecv_base,
        step=0.05,
        cv=cv_n,
        scoring='roc_auc',
        n_jobs=16,
        min_features_to_select=200
    )

    selector.fit(X_train, y_train)

    # Get masks / names of selected features
    selected_mask = selector.support_
    selected_features = X_train.columns[selected_mask].tolist()
    print(f"[RFECV] Selected {len(selected_features)} features out of {X_train.shape[1]}.")

    # Transform splits to selected features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # Optional: monitor CV performance from RFECV
    print(f"[RFECV] Best CV score: {selector.cv_results_['mean_test_score'].max():.3f}")


    # Base estimator (GridSearchCV will override params in param_grid)
    xgb_base = XGBClassifier(
        n_estimators=1000,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=1,
        n_jobs=16
    )

    # CV for GridSearch
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)

    grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=16,
        cv=inner_cv,
        refit=True,
        verbose=1
    )

    grid.fit(X_train_sel, y_train, verbose=False)

    print("Best params from GridSearchCV:")
    print(grid.best_params_)
    print(f"Best CV ROC AUC: {grid.best_score_:.3f}")

    # -----------------------------
    # Evaluate best model on test set
    # -----------------------------
    best_model = grid.best_estimator_

    proba_test = best_model.predict_proba(X_test_sel)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred_test)
    tn, fp, fn, tp = cm.ravel()

    total_One = tn + fp
    total_Two = fn + tp

    acc_one = tn / total_One if total_One > 0 else 0.0
    acc_two = tp / total_Two if total_Two > 0 else 0.0
    auc = roc_auc_score(y_test, proba_test)
    overall_accuracy = (tn + tp) / (total_One + total_Two) if (total_One + total_Two) > 0 else 0.0

    # Store a single row per participant with best hyperparameters
    temp_df = pd.DataFrame({
        "participant_id": [id],
        "total_One": [total_One],
        "total_Two": [total_Two],
        "correct_One": [tn],
        "correct_Two": [tp],
        "accuracy_One": [acc_one],
        "accuracy_Two": [acc_two],
        "overall_accuracy": [overall_accuracy],
        "auc": [auc],
        
        # Optional: also log best hyperparameters
        "best_max_depth": [grid.best_params_["max_depth"]],
        "best_min_child_weight": [grid.best_params_["min_child_weight"]],
        "best_reg_alpha": [grid.best_params_["reg_alpha"]],
        "best_gamma": [grid.best_params_["gamma"]],
        "best_colsample_bytree": [grid.best_params_["colsample_bytree"]],
        "best_colsample_bylevel": [grid.best_params_["colsample_bylevel"]],
        "best_learning_rate": [grid.best_params_["learning_rate"]],
    })

    summary_df = pd.concat([summary_df, temp_df], ignore_index=True)

    print(f"Participant {id} classification report with best params:")
    print(grid.best_params_)
    print(
        f"Accuracy One: {acc_one:.3f}, "
        f"Accuracy Two: {acc_two:.3f}, "
        f"Overall Accuracy: {overall_accuracy:.3f}, "
        f"AUC: {auc:.3f}"
    )


    # Extract top 50 most informative features

    feature_importance = best_model.feature_importances_
    feature_names = X_train_sel.columns.tolist()

    # Create dataframe of feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Get top 50 features
    top_50_features = importance_df.head(50).copy()
    top_50_features['participant_id'] = id
    top_50_features['rank'] = range(1, 51)

    # Append to a master dataframe
    if 'feature_importance_df' not in locals():
        feature_importance_df = pd.DataFrame()

    feature_importance_df = pd.concat([feature_importance_df, top_50_features], ignore_index=True)

    print(f"Top 5 features for participant {id}:")
    print(top_50_features.head())




report_df = (
    summary_df
    .sort_values(['participant_id', 'overall_accuracy'], ascending=[True, False])
    .drop_duplicates(subset='participant_id', keep='first')
)

report_df.to_csv('CNV_ML_3.8_report.csv', index=False)

total_One_final = report_df['total_One'].sum()
total_Two_final = report_df['total_Two'].sum()
correct_One_final = report_df['correct_One'].sum()
correct_Two_final = report_df['correct_Two'].sum()

overall_acc_one = correct_One_final / total_One_final if total_One_final > 0 else 0.0
overall_acc_two = correct_Two_final / total_Two_final if total_Two_final > 0 else 0.0
overall_accuracy = (correct_One_final + correct_Two_final) / (total_One_final + total_Two_final) if (total_One_final + total_Two_final) > 0 else 0.0

print(f'Overall Accuracy One: {overall_acc_one:.3f}')
#0.776

print(f'Overall Accuracy Two: {overall_acc_two:.3f}')
#0.735

print(f'Overall Accuracy: {overall_accuracy:.3f}')
#0.756

# After processing all participants
print("\nFeature importance summary across all participants:")
print(feature_importance_df.head(20))

# Optional: Save to CSV
feature_importance_df.to_csv('top_50_features_per_participant_2.csv', index=False)

# Optional: Find most commonly important features across participants
feature_freq = feature_importance_df.groupby('feature').size().sort_values(ascending=False)
print("\nMost frequently appearing features in top 50:")
print(feature_freq.head(20))