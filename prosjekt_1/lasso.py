# Importing packages.
import numpy as np
from sklearn.linear_model import Lasso

def lasso(alpha, X_train, y_train):
	# Lasso regression
	lasso_model = Lasso(alpha=alpha, fit_intercept=False)
	lasso_model.fit(X_train, y_train)

	return lasso_model
