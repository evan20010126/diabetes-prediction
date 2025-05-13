from sklearn.metrics import accuracy_score
import numpy as np
import numpy as np
from scipy.stats import mode


class SoftVotingEnsemble:

    def __init__(self, estimators, weights=None):
        """
        estimators: list of (name, model) tuples
        weights: list of float, same length as estimators
        """
        self.estimators = estimators
        self.weights = weights if weights is not None else [1.0
                                                            ] * len(estimators)

    def predict_proba(self, X):
        weighted_probas = None

        for (name, model), weight in zip(self.estimators, self.weights):
            proba = model.predict_proba(X) * weight
            if weighted_probas is None:
                weighted_probas = proba
            else:
                weighted_probas += proba

        # Normalize by total weight to get weighted average
        total_weight = sum(self.weights)
        return weighted_probas / total_weight

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class HardVotingEnsemble:

    def __init__(self, estimators):
        """
        estimators: list of (name, model) tuples
        """
        self.estimators = estimators

    def predict(self, X):
        # Collect predictions: shape = (n_models, n_samples)
        preds = np.array([model.predict(X) for _, model in self.estimators])
        # Transpose to shape = (n_samples, n_models)
        preds = preds.T
        # Take mode along each row (i.e., majority vote per sample)
        majority_vote, _ = mode(preds, axis=1, keepdims=False)
        return majority_vote

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def predict_proba(self, X):
        # Collect probabilities: shape = (n_models, n_samples, n_classes)
        probas = np.array(
            [model.predict_proba(X) for _, model in self.estimators])
        # Average probabilities across models
        return np.mean(probas, axis=0)
