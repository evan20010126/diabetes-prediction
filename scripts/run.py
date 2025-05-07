# ---------------------- Imports ----------------------

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import os
import argparse
from omegaconf import OmegaConf
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from cores.data_preprocessing import data_preprocessing
from utils.common import log_args
from utils.logger_util import CustomLogger as logger
from utils.data_utils import split_data, get_features_and_target
from utils.config import SEED

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
    data = data_preprocessing(data=data)

    # 2. Split data
    df_train, df_valid, df_test = split_data(data, **args_dict['data_split'])
    X_train, y_train = get_features_and_target(df_train)
    X_valid, y_valid = get_features_and_target(df_valid)
    X_test, y_test = get_features_and_target(df_test)

    logger.debug(
        f"Train shape: {X_train.shape}; Valid shape: {X_valid.shape}; Test shape: {X_test.shape}"
    )
    logger.debug("Train label distribution: %s",
                 np.unique(y_train, return_counts=True))

    # 3. Training
    for cfg in args.model_cfgs:
        if cfg['name'] == 'TabNetClassifier':
            logger.info(f"Training {cfg['name']}...")
            clf = TabNetClassifier(device_name=device)

            clf.fit(X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    **cfg['training_params'])

            # 4. Testing
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]

            # 5. Evaluataion
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            cm = confusion_matrix(y_test, preds)

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

# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# print("RandomForest acc:", rf.score(X_test, y_test))
