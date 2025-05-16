from utils.logger_util import CustomLogger as logger
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def train_DLmodels(X_train, y_train, X_valid, y_valid, X_test, y_test,
                   model_cfgs, sample_weights, device):
    """
    Train a single TabNetClassifier model and evaluate on the test set.
    Returns: trained model (clf), test set predictions (probs)
    """

    for model_cfg in model_cfgs:

        logger.info(f"Training {model_cfg['name']}...")

        if model_cfg['name'] != 'TabNetClassifier':
            raise NotImplementedError(
                "Only TabNetClassifier is supported in train_model.")

        clf = TabNetClassifier(device_name=device, **model_cfg['params'])
        clf.fit(X_train,
                y_train,
                weights=sample_weights,
                eval_set=[(X_valid, y_valid)],
                **model_cfg['training_params'])

        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        logger.info("\n Evaluation on Test Set:")
        logger.info(f"Accuracy: {round(acc, 4)}; ROC AUC: {round(auc, 4)}")
        logger.info(f"Confusion Matrix:\n{cm}")

    return clf
