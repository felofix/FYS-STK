import numpy as np 
import ffnn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error

def create_design_matrix(x, order):
	order += 1

	X = np.zeros((len(x), order))

	for i in range(order):
		X[:,i] = x**(i)
		
	return X

n = 1000
x = np.random.rand(n)
sorted_inds = np.argsort(x, axis=0).ravel()
y = x*x
X = create_design_matrix(x, 2)
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
ffnn = nn.FFNN(Xtrain, ytrain, epochs=200, n_outputs=1, eta=0.001, n_hidden_layers=2, n_hidden_neurons = 5)
ffnn.train()
ypred = ffnn.predict()
print(mean_squared_error(ytrain, ypred))


"""
# Random seed. 
np.random.seed(0)

# Design matrix.
X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# Outputs. 
xor_gate = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
and_gate = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
or_gate  = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])

ffnn = nn.FFNN(X, xor_gate, eta=0.1,lmbd=0.01, epochs = 1, softmax=True, n_hidden_neurons=2, n_hidden_layers=2)
ffnn.train()
"""