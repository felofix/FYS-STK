import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score

# ensure the same random numbers appear every time
np.random.seed(0)

class FFNN:
    def __init__(self, 
            X_data,
            Y_data,
            n_hidden_neurons = 50,
            n_hidden_layers = 2,
            n_outputs = 1,       # Set to 1 when trying to predict a y. 
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
        self.layers = []

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

        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(self.n_features, self.n_hidden_neurons))
        self.biases.append(np.zeros(self.n_hidden_neurons) + 0.01)

        if self.n_hidden_layers > 1:
            # Hidden layer weights and biases. 
            for i in range(self.n_hidden_layers - 1):
                self.weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons))
                self.biases.append(np.zeros(self.n_hidden_neurons) + 0.01)
        
        self.weights.append(np.random.randn(self.n_hidden_neurons, self.n_outputs))
        self.biases.append(np.zeros(self.n_outputs) + 0.01)

    def forward(self):
        # Moving the neural network forwards.

        # Input layer.
        self.z_h = self.X_data_full                         #@self.input_weights + self.input_bias
        self.activation_layers[0] = self.z_h                #self.sigmoid(self.z_h)

        # Hidden layer.
        for i in range(1, self.n_hidden_layers + 1):
            self.z_h = np.matmul(self.activation_layers[i - 1], self.weights[i - 1]) + self.biases[i - 1]
            self.activation_layers[i] = self.sigmoid(self.z_h)

        # Output layer.
        if self.softmax:
            self.z_o = np.matmul(self.activation_layers[-1], self.weights[-1]) + self.biases[-1]
            exp = np.exp(self.z_o)
            self.probabilities = exp/(np.sum(exp, axis=1, keepdims=True))

        else:
            self.z_o = np.matmul(self.activation_layers[-1], self.output_weights) + self.output_bias

    def backward(self):
        # Moving the neural network backwards. Fix backprop and I'm good. 
    
        if self.softmax: # Categorization. 
            error = self.probabilities - self.Y_data_full

        else: # Regular regression. 
            error = self.z_o - self.Y_data_full.reshape(self.z_o.shape)

        for i in reversed(range(len(self.activation_layers))):
            weight_grad = self.activation_layers[i].T @ error + self.regularization(self.weights[i])
            bias_grad =  np.sum(error, axis=0)
            error = error @ self.weights[i].T * self.activation_layers[i] * (1 - self.activation_layers[i])
            self.weights[i] -= weight_grad
            self.biases[i] -= bias_grad

    def cat_predict(self):
        return np.argmax(self.probabilities, axis=1)

    def sigmoid(self, w):
        # Sigmoid function. 
        return 1/(1 + np.exp(-w))

    def MSE(self, y, y_pred):
        # MSE.
        return mean_squared_error(y, y_pred)

    def R2(self, y, y_pred):
        # R2.
        return r2_score(y, y_pred)

    def regularization(self, w):
        """
        Adds regularization if needed.
        """
        reg = 0

        if self.lmbd > 0.0:
            reg = self.lmbd * w

        return reg

    def error_output(self, output):
        # Does the error output depending on what type of regression we are looking at.
        if self.softmax == True:
            raise NotImplementedError

        error = output - self.Y_data_full.reshape(output.shape)
        return error