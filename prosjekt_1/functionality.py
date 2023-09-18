from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class Franke_Regression:
	def create_design_matrix(self, x, y, order):
	    if (len(x) != len(y)):
	        AssertionError("x and y must have the same length!")

	    number_of_combinations = int((order+1)*(order+2)/2)
	    X = np.zeros((len(x), number_of_combinations))

	    col_nr = 0

	    for i in range(order+1):
	        for j in range(i+1):
	            X[:,col_nr] = x**(i-j)*y**(j)
	            col_nr += 1
	    
	    return X

	def scale(self, X_train, X_test):
	    # Scaling function.
	    scaler = StandardScaler()
	    scaler.fit(X_train)
	    X_train = scaler.transform(X_train)
	    X_test = scaler.transform(X_test)
	    return X_train, X_test

	def find_betas_OLS(self, X, z):
	    return np.linalg.pinv(X.T@X)@X.T@z

	def find_betas_Ridge(self, X, z, lamb):
	    # Calculating beta values with ridge regression. 
	    betas = (np.linalg.pinv(X.T @ X + np.eye(X.shape[1])*lamb) @ X.T) @ z
	    return betas

	def create_lasso(self, alpha, X, y):
		# Lasso regression
		lasso_model = Lasso(alpha=alpha, fit_intercept=False)
		lasso_model.fit(X, y)
		return lasso_model

	def predict_z(self, X, betas):
		return X@betas

	def MSE(self, z, z_pred):
		return 1/len(z)*np.sum((z-z_pred)**2)

	def R2_score(self, z, z_pred):
		z_mean = np.mean(z)
		return 1 - np.sum((z-z_pred)**2)/np.sum((z-z_mean)**2) 

def FrankeFunction(x, y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9* y-7)**2)
	return term1 + term2 + term3 + term4

