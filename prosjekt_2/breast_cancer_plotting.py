import numpy as np 
import ffnn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer
import plotting as p
import activations as act
import tensorffnn as ts

def binary_preprocessing(target_values):
	"""
	Necassery step for the preprocesseing of data. 
	"""

	processed = np.zeros((len(target_values), 2))
	
	for target in range(len(target_values)):
		if target_values[target] == 0:
			processed[target, 0] = 1
		elif target_values[target] == 1:
			processed[target, 1] = 1

	return processed

database = load_breast_cancer()
data = database['data']
targets = database['target']
X = data
y = binary_preprocessing(targets)

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
etas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
lmbds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

accuracies_our = np.zeros((len(etas), len(lmbds)))
accuracies_tsnn = np.zeros((len(etas), len(lmbds)))

for i in range(len(etas)):
	for j in range(len(lmbds)):
		ffnn = nn.FFNN(Xtrain,
					   ytrain,
					   epochs=500, 
					   n_hidden_layers=1, 
					   eta=etas[i],
					   lmbd=lmbds[j], 
					   n_hidden_neurons = 20,
					   softmax=True)

		tsnn = ts.FFNN(n_features = Xtrain.shape[1],  
					  n_hidden_layers=1, 
					  n_hidden_neurons=20, 
					  eta=etas[i],
					  lmbd=lmbds[j],
					  epochs=500,
					  softmax=True)

		ffnn.train(Xtest, ytest)
		tsnn.train(Xtest, ytest)
		accuracies_our[i, j] = np.max(ffnn.accuracies)
		accuracies_tsnn[i, j] = np.max(tsnn.accuracies)

p.plot_heatmap(accuracies_our, etas, lmbds, f'mse_heatmap_Cancer_sigmoid.pdf')
p.plot_heatmap(accuracies_tsnn, etas, lmbds, f'mse_heatmap_Cancer_sigmoid_scikit.pdf')
