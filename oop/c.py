from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import numpy as np

class MyReg(LinearRegression):
    def __init__(self, X, y, pred):
        LinearRegression.__init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
        LinearRegression.fit(self, X, y)
        self.y_pred = LinearRegression.predict(self, pred)
        
data = load_boston()
x = data['data'][:, 0].reshape(506, 1)
y = data['target'].reshape(506,1)

reg = MyReg(X = x, y = y, pred = np.array([5]).reshape(1, 1))
print(reg.y_pred)
