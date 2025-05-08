# ---------------------- Imports ----------------------

# from pathlib import Path
# from matplotlib import pyplot as plt
# import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split

import argparse
from omegaconf import OmegaConf
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from cores.data_preprocessing import impute_missing_values
from utils.data_utils import get_features_and_target

from utils.config import SEED
from utils.common import log_args
from utils.logger_util import CustomLogger as logger

# Models
from cores.Fit_GridSearchCV import fit_random_forest_classifier, fit_tabnet_classifier
from cores.Inference_GridSearchCV import evalutate

from models.tabnet import TabNetSklearnWrapper

# ---------------------- Code ----------------------

# Set the random seed for reproducibility

np.random.seed(SEED)


def main():

    # 0. Configure
    args, args_dict = create_args()
    logger.configure(args.save.out_root)
    log_args(args_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data processing
    data = pd.read_csv(args.dataset_path)
    data = impute_missing_values(data=data)

    # 2. Split data
    df_trainval, df_test = train_test_split(
        data, test_size=args.data_split.test_size, random_state=SEED)

    X_trainval, y_trainval = get_features_and_target(df_trainval)
    X_test, y_test = get_features_and_target(df_test)

    logger.debug(
        f"TrainVal shape: {X_trainval.shape}; Test shape: {X_test.shape}")
    logger.debug("TrainVal label distribution: %s",
                 np.unique(y_trainval, return_counts=True))

    # 3. Fit
    for cfg in args.model_cfgs:

        logger.info(f"Starting {cfg['name']} with pipeline...")
        n_folds = args.data_split.n_folds
        pca_param_grid = args.pca_cfg.param_grid
        model_param_grid = cfg.param_grid

        if cfg['name'] == 'RandomForestClassifier':

            grid_search = fit_random_forest_classifier(
                n_folds=n_folds,
                pca_param_grid=pca_param_grid,
                model_param_grid=model_param_grid,
                X_trainval=X_trainval,
                y_trainval=y_trainval,
            )

            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(
                f"Best cross-validation score: {grid_search.best_score_}")

        elif cfg['name'] == 'TabNetClassifier':

            grid_search = fit_tabnet_classifier(
                n_folds=n_folds,
                pca_param_grid=pca_param_grid,
                model_param_grid=model_param_grid,
                X_trainval=X_trainval,
                y_trainval=y_trainval,
                device=device,
            )

            logger.info(f"Best parameters found: {grid_search.best_params_}")
            logger.info(
                f"Best cross-validation score: {grid_search.best_score_}")

        else:
            ValueError(
                f"Model {cfg.name} is not supported for GridSearchCV with pipeline."
            )

        # Evaluate on the test set
        best_model = grid_search.best_estimator_
        acc, auc, cm = evalutate(best_model, X_test, y_test)
        logger.info("\n Evaluation on Test Set:")
        logger.info(
            f"Accuracy: {round(acc, 4)}; ROC AUC: {round(auc, 4)}\n Confusion Matrix:\n{cm}"
        )


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
