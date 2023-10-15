import numpy
import matplotlib.pyplot as plt
from functionality import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from imageio import imread

# Some random seed. 
np.random.seed(2018)

class Kfolds:

	def __init__(self, model, x, y, z):
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

	def __call__(self, nfolds, degree, alpha=1):
		"""
		-----      Paramters      -----

		- nfolds (str). 
		- degree (str).

		"""

		if self.model == 'OLS':
			return self.OLS(nfolds, degree)

		if self.model == 'Ridge':
			return self.Ridge(nfolds, degree, alpha)

		if self.model == 'Lasso':
			return self.Lasso(nfolds, degree, alpha)

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

	def Ridge(self, nfolds, degree, alpha=1):
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

		mse_KFold = np.zeros(nfolds)

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
			betas = reg.find_betas_Ridge(X_scaled_train, z_train, alpha)
			z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

			mse_KFold[k] = reg.MSE(z_test, z_pred_test)

		mse_KFold = np.mean(mse_KFold)

		return mse_KFold

	def Lasso(self, nfolds, degree, alpha=1):
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

		mse_KFold = np.zeros(nfolds)

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
			betas = reg.find_betas_Lasso(X_scaled_train, z_train, alpha)
			z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

			mse_KFold[k] = reg.MSE(z_test, z_pred_test)

		mse_KFold = np.mean(mse_KFold) 

		return mse_KFold

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

	datatype = 'generated'
	regressiontype = 'OLS'
	if datatype == 'generated':
			# Generate data 
			n = 20
			x = np.linspace(0, 1, n)
			y = np.linspace(0, 1, n)
			x, y = np.meshgrid(x,y)
			z = FrankeFunction(x, y).ravel()
			z+= 0.2*np.random.randn(z.size)

	if datatype == 'real':
		# Load the terrain
		terrain1 = imread('terrain/SRTM_data_Norway_1.tif')[::100, ::100]
		xlen, ylen = terrain1.shape[1], terrain1.shape[0]
		x, y = np.arange(0, xlen), np.arange(0, ylen)
		x, y = np.meshgrid(x, y)
		z = terrain1
		z = (z - np.mean(z)) / np.std(z)

	# Make life easier, flat x and y 
	x_flat = x.flatten()
	y_flat = y.flatten()
	z_flat = z.flatten()

	kfold = Kfolds(regressiontype, x, y, z)
	kfolds = [5, 6, 7, 8, 9, 10]
	degrees = 10
	MSE_ols = np.zeros((degrees, len(kfolds)))

	# OLS. 
	for k in range(len(kfolds)):
		for d in range(degrees):
			MSE_ols[d, k] = kfold(kfolds[k], d + 1, alpha=1)

	for k in range(len(kfolds)):
		plt.plot(np.arange(1, degrees + 1), MSE_ols[:, k], 'o-', label=f'kfolds = {kfolds[k]}')

	plt.legend()
	plt.grid()
	plt.ylabel('MSE')
	plt.xlabel('Polynomial degree')
	plt.savefig(f"plots/crossval_{regressiontype}_{datatype}.pdf")
	plt.show()

	

	

	
	
	










