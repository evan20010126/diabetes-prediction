from sklearn.metrics import accuracy_score
import numpy as np


class SoftVotingEnsemble:

    def __init__(self, estimators):
        """
        estimators: list of (name, model) tuples
        所有 model 必須有 predict_proba()
        """
        self.estimators = estimators

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for _, model in self.estimators]
        return np.mean(probas, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
