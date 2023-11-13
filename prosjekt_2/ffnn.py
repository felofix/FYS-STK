import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tqdm import tqdm
import activations as act


# ensure the same random numbers appear every time
np.random.seed(0)

class FFNN:
    def __init__(self, 
            X_data,
            Y_data,
            n_hidden_neurons = 2,
            n_hidden_layers = 5,
            n_outputs = 2,       
            epochs = 10,
            batch_size = 100,
            eta = 0.1,
            lmbd = 0.0,
            softmax = False,
            activation_function = act.sigmoid,
            opacity = 100,
            verbose = True):
        
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
        self.activation_layers = list(np.zeros(n_hidden_layers   + 1))
        self.layers = []
        self.mse = []
        self.r2 = []
        self.accuracies = []
        self.opacity = opacity
        self.activation_function = activation_function
        self.verbose = verbose

        # Hyperparameters. 
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def train(self, X, y):
        # Trains the neural network for x number of epochs. 

        self.X = self.X_data_full
        self.y = self.Y_data_full

        number_batches = int(self.n_inputs/self.batch_size)
        indeces = np.arange(self.n_inputs)


        if self.verbose:
            for epoch in tqdm(range(self.epochs), desc="Processing epochs"):
                random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
                batches = np.array_split(random_indeces, number_batches)
                
                for b in range(number_batches):
                    self.X = self.X_data_full[batches[b]]
                    self.y = self.Y_data_full[batches[b]]

                    self.forward()
                    self.backward()

                if epoch % self.opacity == 0:
                    if self.softmax:
                        self.accuracies.append(self.accuracy(y, self.cat_predict(self.predict(X))))
                    else:
                        ypred = self.predict(X)
                        self.mse.append(self.MSE(ypred, y))
                        self.r2.append(self.R2(ypred, y))
        else:
            for epoch in range(self.epochs):
                random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
                batches = np.array_split(random_indeces, number_batches)
                
                for b in range(number_batches):
                    self.X = self.X_data_full[batches[b]]
                    self.y = self.Y_data_full[batches[b]]

                    self.forward()
                    self.backward()

                if epoch % self.opacity == 0:
                    if self.softmax:
                        self.accuracies.append(self.accuracy(y, self.cat_predict(self.predict(X))))
                    else:
                        ypred = self.predict(X)
                        self.mse.append(self.MSE(ypred, y))
                        self.r2.append(self.R2(ypred, y))
            
    def create_biases_and_weights(self):
        # Creating biases and weights. 

        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(self.n_features, self.n_hidden_neurons))
        self.biases.append(np.zeros_like(self.n_hidden_neurons, dtype=np.float64) + 0.01)

        if self.n_hidden_layers > 1:
            # Hidden layer weights and biases. 
            for i in range(self.n_hidden_layers - 1):
                self.weights.append(np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons))
                self.biases.append(np.zeros_like(self.n_hidden_neurons, dtype=np.float64) + 0.01)
        
        self.weights.append(np.random.randn(self.n_hidden_neurons, self.n_outputs))
        self.biases.append(np.zeros_like(self.n_hidden_neurons, dtype=np.float64) + 0.01)

    def forward(self):
        # Moving the neural network forwards.
        
        # Input layer.
        self.z_h = self.X                         
        self.activation_layers[0] = self.z_h              

        # Hidden layer.
        for i in range(1, self.n_hidden_layers + 1):
            self.z_h = self.activation_layers[i - 1]@self.weights[i - 1] + self.biases[i - 1]
            self.activation_layers[i] = self.activation_function(self.z_h)

        self.z_o = self.activation_layers[-1]@self.weights[-1] + self.biases[-1]

        # Output layer.
        if self.softmax:
            exp = np.exp(self.z_o)
            self.probabilities = exp/(np.sum(exp, axis=1, keepdims=True))

    def backward(self):
        # Moving the neural network backwards.
        
        if self.softmax: # Categorization. 
            error = self.probabilities - self.y

        else: # Regular regression. 
            error = self.z_o - self.y.reshape(self.z_o.shape)

        # Output.
        weight_grad = self.activation_layers[-1].T @ error + self.regularization(self.weights[-1])
        bias_grad =  np.sum(error, axis=0)
        error = error @ self.weights[-1].T * self.classification(self.activation_layers[-1])
        self.weights[-1] -= weight_grad*self.eta
        self.biases[-1] -= bias_grad*self.eta

        for i in reversed(range(len(self.activation_layers) - 1)):
            weight_grad = self.activation_layers[i].T @ error + self.regularization(self.weights[i])
            bias_grad =  np.sum(error, axis=0)
            self.weights[i] -= weight_grad*self.eta
            self.biases[i] -= bias_grad*self.eta
            error = error @ self.weights[i].T

    def cat_predict(self, probabilities):
        return np.round(probabilities)

    def predict(self, X):
        # Input layer.
        act_layers = list(np.zeros(len(self.activation_layers)))

        z_h = X                        
        act_layers[0] = z_h              

        # Hidden layer.
        for i in range(1, self.n_hidden_layers + 1):
            z_h = act_layers[i - 1]@self.weights[i - 1] + self.biases[i - 1]
            act_layers[i] = self.activation_function(z_h)
            
        z_o = act_layers[-1]@self.weights[-1] + self.biases[-1]

        # Output layer.
        if self.softmax:
            exp = np.exp(z_o)
            probabilities = exp/(np.sum(exp, axis=1, keepdims=True))
            return probabilities

        else:
            return z_o

    def classification(self, w):
        if self.softmax:
            return w * (1 - w)
        else:
            return 1

    def MSE(self, y, y_pred):
        # MSE.
        return mean_squared_error(y, y_pred)

    def R2(self, y, y_pred):
        # R2.
        return r2_score(y, y_pred)

    def accuracy(self, y, y_pred):
        acc = accuracy_score(y, y_pred)
        return acc

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

class LogisticRegression(FFNN):
    def create_biases_and_weights(self):
        # Creating biases and weights. 

        self.weights = []
        self.biases = []

        self.weights.append(np.random.randn(self.n_features, self.n_outputs))
        self.biases.append(np.zeros_like(self.n_outputs, dtype=np.float64) + 0.01)


    def forward(self):
        # Moving the neural network forwards.
        
        # Input layer.
        self.z_h = self.X                         
        self.activation_layers[0] = self.z_h              
        self.z_o = self.activation_function(self.activation_layers[-1]@self.weights[-1] + self.biases[-1])
        self.z_o = self.cat_predict(self.z_o)

    def train(self, X, y):
        # Trains the neural network for x number of epochs. 

        self.X = self.X_data_full
        self.y = self.Y_data_full

        number_batches = int(self.n_inputs/self.batch_size)
        indeces = np.arange(self.n_inputs)


        if self.verbose:
            for epoch in tqdm(range(self.epochs), desc="Processing epochs"):
                random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
                batches = np.array_split(random_indeces, number_batches)
                
                for b in range(number_batches):
                    self.X = self.X_data_full[batches[b]]
                    self.y = self.Y_data_full[batches[b]]

                    self.forward()
                    self.backward()

                if epoch % self.opacity == 0:
                    ypred = self.predict(X)
                    self.accuracies.append(self.accuracy(y, ypred))

        else:
            for epoch in range(self.epochs):
                random_indeces = np.random.choice(indeces, replace=False, size=indeces.size) #shuffle the data
                batches = np.array_split(random_indeces, number_batches)
                
                for b in range(number_batches):
                    self.X = self.X_data_full[batches[b]]
                    self.y = self.Y_data_full[batches[b]]

                    self.forward()
                    self.backward()

                if epoch % self.opacity == 0:
                    ypred = self.predict(X)
                    self.accuracies.append(self.accuracy(y, ypred))

    def predict(self, X):
        # Input layer.
        act_layers = list(np.zeros(len(self.activation_layers)))

        z_h = X                        
        act_layers[0] = z_h              
        z_o = self.activation_function(act_layers[-1]@self.weights[-1] + self.biases[-1])
        return np.round(z_o)


