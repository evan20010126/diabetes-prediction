# ---------------------- Imports ----------------------
import argparse
from omegaconf import OmegaConf
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from cores.data_preprocessing import impute_missing_values_wo_data_leak, add_combined_features, impute_missing_values_with_MICE_wo_data_leak
from utils.common import log_args
from utils.logger_util import CustomLogger as logger
from utils.data_utils import split_data, get_features_and_target
from utils.config import SEED

from sklearn.utils.class_weight import compute_sample_weight

# from models.tabnet import TabNetPretrainedWrapper
from imblearn.combine import SMOTEENN
from utils.importance_analyzation import save_eigenvector_matrix, compute_importance, save_importance, compute_eigenvector_matrix_with_weighted

from cores.train_DL import train_DLmodels
from cores.tune_ensemble import tune_ensemble

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

    # Split data
    df_train, df_valid, df_test = split_data(data, **args_dict['data_split'])

    # Override test dataset if specified
    if args.test_dataset != "":
        df_test = pd.read_csv(args.test_dataset)  # override

    # Impute missing values
    if args.imputation.strategy == 'MICE':

        df_train, df_valid, df_test = impute_missing_values_with_MICE_wo_data_leak(
            df_train,
            df_valid,
            df_test,
            target_cols=list(args.imputation.target_cols),
            ignore_cols=list(args.imputation.ignore_cols),
            max_iter=args.imputation.max_iter,
            seed=SEED)

    elif args.imputation.strategy == 'dynamic_imputer':

        df_train, df_valid, df_test = impute_missing_values_wo_data_leak(
            df_train, df_valid, df_test)

    # Feature generation
    if args.feature_generation == "combined_features":
        df_train = add_combined_features(df_train)
        df_valid = add_combined_features(df_valid)
        df_test = add_combined_features(df_test)

    X_train, y_train = get_features_and_target(df_train)

    # Sampling
    if args.sampling == "SMOTEENN":
        smoteen = SMOTEENN(random_state=SEED)
        X_train, y_train = smoteen.fit_resample(X_train, y_train)

    X_valid, y_valid = get_features_and_target(df_valid)
    X_test, y_test = get_features_and_target(df_test)

    logger.debug(
        f"Train shape: {X_train.shape}; Valid shape: {X_valid.shape}; Test shape: {X_test.shape}"
    )
    logger.debug("Train label distribution: %s",
                 np.unique(y_train, return_counts=True))

    # Feature compression
    if args.feature_compression.strategy == 'PCA':

        # Standardize data before PCA
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        n_components = args.feature_compression.n_components
        logger.info(
            f"Applying StandarScaler & PCA with n_components={n_components}..."
        )
        pca = PCA(n_components=n_components, random_state=SEED)
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

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train) if args.sample_weights.strategy == "balanced" else 0

    # Training
    # TODO: Make trained model as a dict, and input into ensemble learning
    clf = train_DLmodels(X_train=X_train,
                         y_train=y_train,
                         X_valid=X_valid,
                         y_valid=y_valid,
                         X_test=X_test,
                         y_test=y_test,
                         model_cfgs=args['training'],
                         sample_weights=sample_weights,
                         device=device)

    weighted_eigenvector_matrix = compute_eigenvector_matrix_with_weighted(
        eigenvector_matrix, clf.feature_importances_)
    importance = compute_importance(
        eigenvector_matrix=weighted_eigenvector_matrix)
    save_importance(importance=importance,
                    feature_names=df_train.columns[:-1],
                    out_root=args.save.out_root,
                    title="Weighted_PCA_Feature_Importance")
    # Save eigenvector matrix
    save_eigenvector_matrix(eigenvector_matrix=weighted_eigenvector_matrix,
                            feature_names_before_PCA=df_train.columns[:-1],
                            out_root=args.save.out_root,
                            title="Weighted_PCA_Eigenvector_Matrix")

    # Ensemble
    X_trainval = np.concatenate((X_train, X_valid), axis=0)
    y_trainval = np.concatenate((y_train, y_valid), axis=0)

    best_ensemble = tune_ensemble(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        X_test=X_test,
        y_test=y_test,
        clf=clf,
        ensemble_learning_cfg=args['ensemble_learning'])

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
