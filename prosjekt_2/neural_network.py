import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score

class FNNN:

	def __init__(self, 
			X_data,
			Y_data,
			n_hidden_neurons = 50,
			n_hidden_layers = 2,
			n_outputs = 1,			# Set to 1 when trying to predict a y. 
			epochs = 10,
			batch_size = 100,
			eta = 0.1,
			lmbd = 0.0,
			softmax = False):
		
		# Data. 
		self.X_data_full = X_data
		self.Y_data_full = Y_data

		# Internal info. 
		self.n_inputs = X_data.shape[0]
		self.n_features = X_data.shape[1]
		self.n_hidden_layers = n_hidden_layers
		self.n_hidden_neurons = n_hidden_neurons
		self.n_outputs = n_outputs
		self.softmax = softmax
		self.activation_layers = np.zeros((n_hidden_layers + 1, self.n_inputs, self.n_hidden_neurons))

		# Hyperparameters. 
		self.epochs = epochs
		self.batch_size = batch_size
		self.iterations = self.n_inputs // self.batch_size
		self.eta = eta
		self.lmbd = lmbd

		self.create_biases_and_weights()

	def train(self):
		# Trains the neural network for x number of epochs. 

		for epoch in range(self.epochs):
			self.forward()
			self.backward()

	def create_biases_and_weights(self):
		# Creating biases and weights. 

		# Input weights and biases. 
		self.input_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
		self.input_bias = np.zeros(self.n_hidden_neurons) + 0.01

		# Hidden layer weights and biases. 
		self.hidden_weights = np.array([np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons) for i in range(self.n_hidden_layers)])  # For all hidden layers. 
		self.hidden_bias = np.array([np.zeros(self.n_hidden_neurons) + 0.01 for i in range(self.n_hidden_layers)])					 		  # For all hidden layers. 

		# Output layer weights and biases. 
		self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_outputs)
		self.output_bias = np.zeros(self.n_outputs) + 0.01

	def forward(self):
		# Moving the neural network forwards. 

		# Input layer.
		self.z_h = np.matmul(self.X_data_full, self.input_weights) + self.input_bias
		self.activation_layers[0] = self.sigmoid(self.z_h)
		
		# Hidden layer.
		for i in range(1, self.n_hidden_layers):
			self.z_h = np.matmul(self.activation_layers[i - 1], self.hidden_weights[i]) + self.hidden_bias[i]
			self.activation_layers[i] = self.sigmoid(self.z_h)

		# Output layer.
		self.z_o = np.matmul(self.activation_layers[-1], self.output_weights) + self.output_bias

		print(self.z_o[0])

	def backward(self):
		# Moving the neural network backwards.
    
	    # Output layer.
	    error = self.Y_data_full.reshape(self.z_o.shape) - self.z_o 
	    self.output_weights_gradient = np.matmul(self.activation_layers[-1].T, error)
	    self.output_bias_gradient = np.sum(error, axis=0)

	    if self.lmbd > 0.0:
	        self.output_weights_gradient += self.lmbd * self.output_weights

	    self.output_weights -= self.eta * self.output_weights_gradient
	    self.output_bias -= self.eta * self.output_bias_gradient
	    
	    # Hidden layers.
	    for i in reversed(range(1, self.n_hidden_layers)):
	        error = np.matmul(error, self.output_weights.T) * self.activation_layers[i] * (1 - self.activation_layers[i])

	        self.hidden_weights_gradient = np.matmul(self.activation_layers[i-1].T, error)
	        self.hidden_bias_gradient = np.sum(error, axis=0)
	        
	        if self.lmbd > 0.0:
	            self.hidden_weights_gradient += self.lmbd * self.hidden_weights[i]
	        
	        self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient
	        self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient


	    # Propagate error back to the first hidden layer from the output.
	    error = np.matmul(error, self.hidden_weights[0].T) * self.activation_layers[0] * (1 - self.activation_layers[0])
	    
	    # Input layer
	    self.input_weight_gradient = np.matmul(self.X_data_full.T, error)
	    self.input_bias_gradient = np.sum(error, axis=0)

	    if self.lmbd > 0.0:
	        self.input_weight_gradient += self.lmbd * self.input_weights

	    self.input_weights -= self.eta * self.input_weight_gradient
	    self.input_bias -= self.eta * self.input_bias_gradient

	def predict(self):
		# Input layer.
		z_h = np.matmul(self.X_data_full, self.input_weights) + self.input_bias
		activation = self.sigmoid(self.z_h)
		
		# Hidden layer.
		for i in range(1, self.n_hidden_layers):
			z_h = np.matmul(activation, self.hidden_weights[i]) + self.hidden_bias[i]
			activation = self.sigmoid(self.z_h)
		
		# Output layer.
		z_o = np.matmul(activation, self.output_weights) + self.output_bias

		return z_o

	def sigmoid(self, w):
		# Sigmoid function. 
		return 1/(1 + np.exp(-w))

	def MSE(self, y, y_pred):
		# MSE.
		return mean_squared_error(y, y_pred)

	def R2(self, y, y_pred):
		# R2.
		return r2_score(y, y_pred)

	def error_output(self, output):
		# Does the error output depending on what type of regression we are looking at.
		if self.softmax == True:
			raise NotImplementedError

		error = output - self.Y_data_full.reshape(output.shape)
		return error



	