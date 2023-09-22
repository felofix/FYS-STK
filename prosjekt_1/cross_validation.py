import numpy
import matplotlib.pyplot as plt
from functionality import *

# Some random seed. 
np.random.seed(14)

class Kfold:

	def __init__(self, model, x, y, z, alphas):
		"""
		-----      Paramters      -----

		- Model (str). 'OLS', 'Ridge' or 'Lasso'.
		- x. (array). 
		- y. (array).
		- z. (array).
		- alphas. (array).

		"""

		assert model in ['OLS', 'Ridge', 'Lasso']
		self.model = model	
		self.x = x.flatten()
		self.y = y.flatten()
		self.z = z.flatten()
		self.alphas = alphas

	def __call__(self, nfolds, degree):
		"""
		-----      Paramters      -----

		- nfolds (str). 
		- degree (str).

		"""

		if self.model == 'OLS':
			return self.OLS(nfolds, degree)

		if self.model == 'Ridge':
			return self.Ridge(nfolds, degree)

		if self.model == 'Lasso':
			return self.Lasso(nfolds, degree)

	def OLS(self, nfolds, degree):
		"""
		-----      Paramters      -----

		- nfolds (str). 
		- degree (str).

		"""

		# Shuffeling dataset
		np.random.shuffle(self.x)
		np.random.shuffle(self.y)
		np.random.shuffle(self.z)

		mse_KFold = np.zeros(nfolds)

		for k in range(nfolds):
				# Splitting data into k equal parts.
				x_split = np.array(np.array_split(self.x, nfolds))
				y_split = np.array(np.array_split(self.y, nfolds))
				z_split = np.array(np.array_split(self.z, nfolds))

				# Training data.
				x_train = numpy.concatenate(x_split[np.arange(nfolds) != k])
				y_train = numpy.concatenate(y_split[np.arange(nfolds) != k])
				z_train = numpy.concatenate(z_split[np.arange(nfolds) != k])

				# Test data. 
				x_test = x_split[k]
				y_test = y_split[k]
				z_test = z_split[k]

				# Create coefficients.
				reg = Franke_Regression()
				X_train = reg.create_design_matrix(x_train, y_train, degree)
				X_test = reg.create_design_matrix(x_test, y_test, degree)
				X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled. 

				# Training.
				betas = reg.find_betas_OLS(X_scaled_train, z_train)
				z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

				mse_KFold[k] = reg.MSE(z_test, z_pred_test)

		return np.mean(mse_KFold)

	def Ridge(self, nfolds, degree):
		"""
		-----      Paramters      -----

		- nfolds (str). 
		- degree (str).

		"""

		# Shuffeling dataset
		np.random.shuffle(self.x)
		np.random.shuffle(self.y)
		np.random.shuffle(self.z)

		mse_KFold = np.zeros((len(self.alphas), nfolds))

		for a in range(len(alphas)):
			for k in range(nfolds):
				# Splitting data into k equal parts.
				x_split = np.array(np.array_split(self.x, nfolds))
				y_split = np.array(np.array_split(self.y, nfolds))
				z_split = np.array(np.array_split(self.z, nfolds))

				# Training data.
				x_train = numpy.concatenate(x_split[np.arange(nfolds) != k])
				y_train = numpy.concatenate(y_split[np.arange(nfolds) != k])
				z_train = numpy.concatenate(z_split[np.arange(nfolds) != k])

				# Test data. 
				x_test = x_split[k]
				y_test = y_split[k]
				z_test = z_split[k]

				# Create coefficients.
				reg = Franke_Regression()
				X_train = reg.create_design_matrix(x_train, y_train, degree)
				X_test = reg.create_design_matrix(x_test, y_test, degree)
				X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled. 

				# Training.
				betas = reg.find_betas_Ridge(X_scaled_train, z_train, self.alphas[a])
				z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

				mse_KFold[a, k] = reg.MSE(z_test, z_pred_test)

		mse_KFold = np.mean(mse_KFold, axis = 1)

		return mse_KFold

	def Lasso(self, nfolds, degree):
		"""
		-----      Paramters      -----

		- nfolds (str). 
		- degree (str).

		"""

		# Shuffeling dataset
		np.random.shuffle(self.x)
		np.random.shuffle(self.y)
		np.random.shuffle(self.z)

		mse_KFold = np.zeros((len(self.alphas), nfolds))

		for a in range(len(alphas)):
			for k in range(nfolds):
				# Splitting data into k equal parts.
				x_split = np.array(np.array_split(self.x, nfolds))
				y_split = np.array(np.array_split(self.y, nfolds))
				z_split = np.array(np.array_split(self.z, nfolds))

				# Training data.
				x_train = numpy.concatenate(x_split[np.arange(nfolds) != k])
				y_train = numpy.concatenate(y_split[np.arange(nfolds) != k])
				z_train = numpy.concatenate(z_split[np.arange(nfolds) != k])

				# Test data. 
				x_test = x_split[k]
				y_test = y_split[k]
				z_test = z_split[k]

				# Create coefficients.
				reg = Franke_Regression()
				X_train = reg.create_design_matrix(x_train, y_train, degree)
				X_test = reg.create_design_matrix(x_test, y_test, degree)
				X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled.

				# Training.
				betas = reg.find_betas_Lasso(X_scaled_train, z_train, self.alphas[a])
				z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

				mse_KFold[a, k] = reg.MSE(z_test, z_pred_test)

		mse_KFold = np.mean(mse_KFold, axis = 1) 


if __name__ == "__main__":
	### TEST ###
	# Generate data 
	n = 100
	x = np.sort(np.random.uniform(0, 1, n))
	y = np.sort(np.random.uniform(0, 1, n))
	x, y = np.meshgrid(x,y)
	z = FrankeFunction(x, y) + np.random.normal(scale = 1, size = (n,n))
	alphas = [0.0001, 0.001, 0.01, 0.1, 1]
	
	kfold = Kfold('OLS', x, y, z, alphas)
	mse = kfold(10, 5)

	
	
	










