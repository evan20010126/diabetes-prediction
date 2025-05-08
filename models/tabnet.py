from sklearn.base import BaseEstimator, ClassifierMixin
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score


class TabNetSklearnWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, n_d=8, n_steps=3, seed=0, device_name='auto'):

        self.n_d = n_d  # = n_a
        self.n_steps = n_steps
        self.seed = seed
        self.device_name = device_name

        self.model = None

    def fit(self, X, y):

        self.model = TabNetClassifier(n_d=self.n_d,
                                      n_a=self.n_d,
                                      n_steps=self.n_steps,
                                      seed=self.seed,
                                      device_name=self.device_name)
        self.model.fit(X, y, max_epochs=1000, patience=50, batch_size=128)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
