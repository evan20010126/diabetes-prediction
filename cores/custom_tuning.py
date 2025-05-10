from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from models.ensemble import SoftVotingEnsemble, HardVotingEnsemble


def grid_search_ensemble(X,
                         y,
                         tabnet_model,
                         model_dict,
                         param_grid,
                         n_splits=5,
                         scoring='accuracy',
                         mode='soft'):
    """
    Parameters
    ----------
    X, y : ndarray
    tabnet_model : fixed pretrained TabNet
    model_dict : dict[str, estimator]
        Example: {'knn': KNN(), 'etc': ExtraTreesClassifier()}
    param_grid : dict[str, list]
        Example: {'knn__n_neighbors': [...], 'etc__n_estimators': [...]}
    n_splits : int
    scoring : str, one of {'accuracy', 'roc_auc'}

    Returns
    -------
    best_result : dict
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = -np.inf
    best_result = None

    # Generate all combinations of parameters
    all_param_sets = list(ParameterGrid(param_grid))

    for i, param_set in enumerate(all_param_sets):

        scores = []

        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Prepare models for this fold
            trained_models = []

            for name, base_model in model_dict.items():
                model = clone(base_model)
                model_params = {
                    key.split('__')[1]: val
                    for key, val in param_set.items()
                    if key.startswith(f"{name}__")
                }
                model.set_params(**model_params)
                model.fit(X_train, y_train)
                trained_models.append((name, model))

            # 組合 ensemble
            if mode == 'soft':
                ensemble = SoftVotingEnsemble([('tabnet', tabnet_model)] +
                                              trained_models)
            elif mode == 'hard':
                ensemble = HardVotingEnsemble([('tabnet', tabnet_model)] +
                                              trained_models)

            y_pred = ensemble.predict(X_valid)
            if scoring == 'accuracy':
                score = accuracy_score(y_valid, y_pred)
            elif scoring == 'roc_auc':
                y_prob = ensemble.predict_proba(X_valid)[:, 1]
                score = roc_auc_score(y_valid, y_prob)
            else:
                raise ValueError(f"Unsupported scoring: {scoring}")

            scores.append(score)

        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_result = {"params": param_set, "score": avg_score}

        print(
            f"[{i+1}/{len(all_param_sets)}] Params={param_set}, {scoring}: {avg_score:.4f}"
        )

    # 重新訓練最佳模型組合，用全體資料
    final_models = []
    for name, base_model in model_dict.items():
        model = clone(base_model)
        best_params = {
            key.split('__')[1]: val
            for key, val in best_result['params'].items()
            if key.startswith(f"{name}__")
        }
        model.set_params(**best_params)
        model.fit(X, y)
        final_models.append((name, model))

    # 組合最佳的 ensemble
    if mode == 'soft':
        best_ensemble = SoftVotingEnsemble([('tabnet', tabnet_model)] +
                                           final_models,
                                           weights=[0.5, 0.25, 0.25])
    elif mode == 'hard':
        best_ensemble = HardVotingEnsemble([('tabnet', tabnet_model)] +
                                           final_models)

    return best_ensemble, {
        'best_params': best_result['params'],
        'best_score': best_result['score'],
    }
