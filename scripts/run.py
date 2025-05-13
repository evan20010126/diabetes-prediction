# ---------------------- Imports ----------------------

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier

import os
import argparse
from omegaconf import OmegaConf
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cores.data_preprocessing import impute_missing_values, impute_missing_values_wo_data_leak, add_combined_features, impute_missing_values_with_MICE, impute_missing_values_with_MICE_wo_data_leak

from utils.common import log_args
from utils.logger_util import CustomLogger as logger
from utils.data_utils import split_data, get_features_and_target
from utils.config import SEED

from sklearn.utils.class_weight import compute_sample_weight

# from models.tabnet import TabNetPretrainedWrapper
from models.ensemble import SoftVotingEnsemble
from cores.custom_tuning import grid_search_soft_ensemble
from imblearn.combine import SMOTEENN
import seaborn as sns
from utils.importance_analyzation import save_eigenvector_matrix, compute_importance, save_importance, compute_eigenvector_matrix_with_weighted

# ---------------------- Code ----------------------

# Set the random seed for reproducibility

np.random.seed(SEED)


def main():

    # Configure
    args, args_dict = create_args()
    logger.configure(args.save.out_root)
    log_args(args_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data processing
    data = pd.read_csv(args.dataset_path)
    """
    # data = impute_missing_values_with_MICE(data=data)
    target_cols = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]  # 只填補這些欄位
    ignore_cols = ["Outcome"]  # 排除 label 不作為解釋變數

    data = impute_missing_values_with_MICE(data,
                                           target_cols=target_cols,
                                           ignore_cols=ignore_cols,
                                           max_iter=1000,
                                           seed=SEED)
    """

    # Split data
    df_train, df_valid, df_test = split_data(data, **args_dict['data_split'])

    # df_test = pd.read_csv("data/Frankfurt_Hospital_diabetes.csv")
    # df_train, df_valid, df_test = impute_missing_values_wo_data_leak(df_train, df_valid, df_test)
    fix_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    df_train, df_valid, df_test = impute_missing_values_with_MICE_wo_data_leak(
        df_train,
        df_valid,
        df_test,
        target_cols=fix_cols,
        ignore_cols=['Outcome'],
        max_iter=1000,
        seed=SEED)

    df_train = add_combined_features(df_train)
    df_valid = add_combined_features(df_valid)
    df_test = add_combined_features(df_test)
    # Check nan

    X_train, y_train = get_features_and_target(df_train)

    smoteen = SMOTEENN(random_state=SEED)
    X_train, y_train = smoteen.fit_resample(X_train, y_train)

    X_valid, y_valid = get_features_and_target(df_valid)
    X_test, y_test = get_features_and_target(df_test)

    logger.debug(
        f"Train shape: {X_train.shape}; Valid shape: {X_valid.shape}; Test shape: {X_test.shape}"
    )
    logger.debug("Train label distribution: %s",
                 np.unique(y_train, return_counts=True))

    if 'pca_cfg' in args_dict:

        # Standardize data before PCA
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        pca_cfg = args.pca_cfg
        logger.info(
            f"Applying PCA with n_components={pca_cfg['n_components']}...")
        pca = PCA(n_components=pca_cfg['n_components'], random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_valid = pca.transform(X_valid)
        X_test = pca.transform(X_test)

        logger.debug(
            f"After PCA, Train shape: {X_train.shape}; Valid shape: {X_valid.shape}; Test shape: {X_test.shape}"
        )

        # Importance analysis
        eigenvector_matrix = pca.components_.T  # shape: (n_features, n_components)
        importance = compute_importance(eigenvector_matrix=eigenvector_matrix)
        save_eigenvector_matrix(eigenvector_matrix=eigenvector_matrix,
                                feature_names_before_PCA=df_train.columns[:-1],
                                out_root=args.save.out_root,
                                title="PCA_Eigenvector_Matrix")
        save_importance(importance=importance,
                        feature_names=df_train.columns[:-1],
                        out_root=args.save.out_root,
                        title="PCA_Feature_Importance")

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Training
    for cfg in args.model_cfgs:
        if cfg['name'] == 'TabNetClassifier':
            logger.info(f"Training {cfg['name']}...")
            clf = TabNetClassifier(device_name=device)

            clf.fit(X_train,
                    y_train,
                    weights=sample_weights,
                    eval_set=[(X_valid, y_valid)],
                    **cfg['training_params'])

            # Testing
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]

            # Evaluataion
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            cm = confusion_matrix(y_test, preds)

            logger.info("\n Evaluation on Test Set:")
            logger.info(
                f"Accuracy: {round(acc, 4)}; ROC AUC: {round(auc, 4)}\n Confusion Matrix:\n{cm}"
            )

            weighted_eigenvector_matrix = compute_eigenvector_matrix_with_weighted(
                eigenvector_matrix, clf.feature_importances_)
            importance = compute_importance(
                eigenvector_matrix=weighted_eigenvector_matrix)
            save_importance(importance=importance,
                            feature_names=df_train.columns[:-1],
                            out_root=args.save.out_root,
                            title="Weighted_PCA_Feature_Importance")
            # Save eigenvector matrix
            save_eigenvector_matrix(
                eigenvector_matrix=weighted_eigenvector_matrix,
                feature_names_before_PCA=df_train.columns[:-1],
                out_root=args.save.out_root,
                title="Weighted_PCA_Eigenvector_Matrix")

            # Ensemble
            model_dict = {
                'knn': KNeighborsClassifier(),
                'etc': ExtraTreesClassifier(),
                # 'xgb':
                # XGBClassifier(use_label_encoder=False,
                #   eval_metric='logloss',
                #   random_state=42)
            }

            param_grid = {
                'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                'knn__weights': ['uniform', 'distance'],
                'etc__n_estimators': [100, 200],
                'etc__max_depth': [None, 10],
                # 'xgb__n_estimators': [100, 200],
                # 'xgb__max_depth': [3, 6]
            }

            X_trainval = np.concatenate((X_train, X_valid), axis=0)
            y_trainval = np.concatenate((y_train, y_valid), axis=0)

            best_ensemble, _ = grid_search_soft_ensemble(
                X=X_trainval,
                y=y_trainval,
                tabnet_model=clf,
                model_dict=model_dict,
                param_grid=param_grid,
                n_splits=5,
                scoring='accuracy'  # or 'roc_auc'
            )

            y_pred = best_ensemble.predict(X_test)
            y_proba = best_ensemble.predict_proba(X_test)[:, 1]  # 若為二分類
            acc = best_ensemble.score(X_test, y_test)
            logger.info("Ensemble Accuracy:", acc)
            logger.info("Ensemble ROC AUC:", roc_auc_score(y_test, y_proba))
            logger.info("Ensemble Confusion Matrix:",
                        confusion_matrix(y_test, y_pred))

            # # for name, model in ensemble.estimators:
            # # acc = accuracy_score(y_test, model.predict(X_test))
            # # print(f"{name} Accuracy: {acc:.4f}")
            # exit()


def create_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        help='Path to the config file',
        default='configs/run_tabnet.yml',
        required=True,
    )
    parser.add_argument('--dataset_path',
                        type=str,
                        help='E.g. ./data/Pima_Indians_diabetes.csv',
                        required=True)

    ARGS = parser.parse_args()
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    base = OmegaConf.load(ARGS.config)
    override = OmegaConf.create({
        'dataset_path': ARGS.dataset_path,
        'save': {
            'out_root': f"{base.save.out_root}/{current_time}"
        }
    })
    cfg = OmegaConf.merge(base, override)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    return cfg, cfg_dict


if __name__ == "__main__":
    main()

# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# print("RandomForest acc:", rf.score(X_test, y_test))
