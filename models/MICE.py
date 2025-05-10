from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class ScaledRegressor(BaseEstimator, RegressorMixin):
    """Linear regressor with internal standard scaling of X."""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
