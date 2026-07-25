import numpy as np

class MyLogisticRegression():
    def __init__(self, learning_rate = 0.1, n_iter = 1000):
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.n_iter = n_iter
    def _sigmoid(self, z ):
        z = np.clip(z, -500, 500) # защита от переполнеия
        return (1/ (1 + np.exp(-z)))
    
    def fit(self, X, y):
        self.b = 0
        N = X.shape[0] 
        self.w = [0] * X.shape[1]
        X = np.array(X)
        y = np.array(y)
        for i in range(self.n_iter):
            z = X @ self.w + self.b 
            p = self._sigmoid(z)
            error = p - y
            dw = ( 1 / N ) * X.T @ error# от слова  differential ( производнаа)
            db = ( 1 / N) * np.sum(error)
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
        return self
    
    def predict_proba(self, X):
        X = np.array(X)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        p0 = 1 - p1
        arr = np.column_stack((p0, p1))
        return arr

    def predict(self, X, threshold = 0.5):
        probas = self.predict_proba(X)
        prediction = (probas[:, 1] >= threshold).astype('int')

        return prediction


