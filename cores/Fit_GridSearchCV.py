from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils.config import SEED
from utils.logger_util import CustomLogger as logger

from models.tabnet import TabNetSklearnWrapper

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('pca', PCA()),  # Apply PCA
    ('classifier', RandomForestClassifier(random_state=SEED))  # Classifier
])


def fit_random_forest_classifier(n_folds, pca_param_grid, model_param_grid,
                                 X_trainval, y_trainval):

    param_grid = {
        **{
            f"pca__{k}": v
            for k, v in pca_param_grid.items()
        },
        **{
            f"classifier__{k}": v
            for k, v in model_param_grid.items()
        }
    }

    logger.debug(
        f"Hyperparameter tuning with GridSearchCV for RandomForestClassifier..."
    )
    logger.debug(f"Pipeline: {param_grid}")

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=n_folds,
                               scoring='accuracy',
                               verbose=0,
                               n_jobs=-1)

    grid_search.fit(X_trainval, y_trainval)

    return grid_search


def fit_tabnet_classifier(n_folds, pca_param_grid, model_param_grid,
                          X_trainval, y_trainval, device):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('pca', PCA()),  # Apply PCA
        ('classifier', TabNetSklearnWrapper(device_name=device,
                                            seed=SEED))  # Classifier
    ])

    param_grid = {
        **{
            f"pca__{k}": v
            for k, v in pca_param_grid.items()
        },
        **{
            f"classifier__{k}": v
            for k, v in model_param_grid.items()
        }
    }

    logger.debug(
        f"Hyperparameter tuning with GridSearchCV for TabNetClassifier...")
    logger.debug(f"Pipeline: {param_grid}")

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=n_folds,
                               scoring='accuracy',
                               verbose=0,
                               n_jobs=-1)

    grid_search.fit(X_trainval, y_trainval)

    # TODO: How to save the model after training?
    # TODO: Add valid data during training tabnet. Maybe can use ParamGridSearchCV

    return grid_search
