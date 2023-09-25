import numpy
import matplotlib.pyplot as plt
from functionality import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Some random seed. 
np.random.seed(14)

class Kfolds:

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
		data = np.column_stack((self.x, self.y, self.z))
		np.random.shuffle(data)
		self.x, self.y, self.z = data[:, 0], data[:, 1], data[:, 2]

		mse_KFold = np.zeros(nfolds)

		length = len(self.x)
		divisible_length = (length // nfolds) * nfolds

		self.x = self.x[:divisible_length]
		self.y = self.y[:divisible_length]
		self.z = self.z[:divisible_length]

		# Splitting data into k equal parts.
		x_split = np.array(np.array_split(self.x, nfolds))
		y_split = np.array(np.array_split(self.y, nfolds))
		z_split = np.array(np.array_split(self.z, nfolds))

		for k in range(nfolds):
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
		data = np.column_stack((self.x, self.y, self.z))
		np.random.shuffle(data)
		self.x, self.y, self.z = data[:, 0], data[:, 1], data[:, 2]

		mse_KFold = np.zeros(nfolds)

		length = len(self.x)
		divisible_length = (length // nfolds) * nfolds

		self.x = self.x[:divisible_length]
		self.y = self.y[:divisible_length]
		self.z = self.z[:divisible_length]

		# Splitting data into k equal parts.
		x_split = np.array(np.array_split(self.x, nfolds))
		y_split = np.array(np.array_split(self.y, nfolds))
		z_split = np.array(np.array_split(self.z, nfolds))

		mse_KFold = np.zeros((len(self.alphas), nfolds))

		for a in range(len(alphas)):
			for k in range(nfolds):
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
		data = np.column_stack((self.x, self.y, self.z))
		np.random.shuffle(data)
		self.x, self.y, self.z = data[:, 0], data[:, 1], data[:, 2]

		mse_KFold = np.zeros(nfolds)

		length = len(self.x)
		divisible_length = (length // nfolds) * nfolds

		self.x = self.x[:divisible_length]
		self.y = self.y[:divisible_length]
		self.z = self.z[:divisible_length]

		# Splitting data into k equal parts.
		x_split = np.array(np.array_split(self.x, nfolds))
		y_split = np.array(np.array_split(self.y, nfolds))
		z_split = np.array(np.array_split(self.z, nfolds))

		mse_KFold = np.zeros((len(self.alphas), nfolds))

		for a in range(len(alphas)):
			for k in range(nfolds):	
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

	def kfold_with_sklearn(self, nfolds, degree):
	    # Concatenate data and shuffle
		x, y, z = self.x, self.y, self.z
		data = np.column_stack((x, y, z))
		np.random.shuffle(data)
		x, y, z = data[:, 0], data[:, 1], data[:, 2]
	    
		# Prepare KFold
		kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
	    
		mse_values = []
	    
		for train_index, test_index in kf.split(x):
	        # Split data
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			z_train, z_test = z[train_index], z[test_index]
	        
	        # Create polynomial features
			poly = PolynomialFeatures(degree=degree)
			X_train = poly.fit_transform(np.column_stack((x_train, y_train)))
			X_test = poly.transform(np.column_stack((x_test, y_test)))
	        
	        # Scale data
			scaler = StandardScaler()
			X_train = scaler.fit_transform(X_train)
			X_test = scaler.transform(X_test)
	        
	        # Train model
			model = LinearRegression().fit(X_train, z_train)
	        
	        # Predict
			z_pred = model.predict(X_test)
	        
	        # Compute MSE
			mse_values.append(mean_squared_error(z_test, z_pred))
	    
		return np.mean(mse_values)

if __name__ == "__main__":
	### TEST ###
	# Generate data 
	#
	np.random.seed(14)

	# Generate data 
	n = 40
	x = np.linspace(0, 1, n)
	y = np.linspace(0, 1, n)
	x, y = np.meshgrid(x,y)
	z = FrankeFunction(x, y).ravel()
	z+= 0.2*np.random.randn(z.size)
	alphas =[0.0001, 0.001, 0.01, 0.1, 1]
	kfold = Kfolds('OLS', x, y, z, alphas)
	kfolds = [5, 6, 7, 8, 9, 10]
	degrees = 10
	MSE_ols = np.zeros((degrees, len(kfolds)))

	# OLS. 
	for k in range(len(kfolds)):
		for d in range(degrees):
			MSE_ols[d, k] = kfold(kfolds[k], d + 1)

	for k in range(len(kfolds)):
		plt.plot(np.arange(1, degrees + 1), MSE_ols[:, k], 'o-', label=f'kfolds = {kfolds[k]}')

	plt.legend()
	plt.grid()
	plt.show()

	

	

	
	
	










