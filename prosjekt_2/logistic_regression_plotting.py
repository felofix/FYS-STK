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

database = load_breast_cancer()
data = database['data']
targets = database['target']
X = data
y = targets

scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
etas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
lmbds = [1e-4, 1e-3, 1e-2, 1e-1, 1]

accuracies_our = np.zeros((len(etas), len(lmbds)))
accuracies_tsnn = np.zeros((len(etas), len(lmbds)))

for i in range(len(etas)):
	for j in range(len(lmbds)):
		logreg = nn.LogisticRegression(Xtrain,
					   ytrain,
					   epochs=2000, 
					   n_hidden_layers=0, 
					   n_outputs = 1,
					   eta=etas[i],
					   lmbd=lmbds[j], 
					   n_hidden_neurons = 2,
					   softmax=False)
		
		tsnn = ts.LogisticRegression(n_features = Xtrain.shape[1],  
					  n_hidden_layers=0, 
					  n_hidden_neurons=2, 
					  n_outputs=1,
					  eta=etas[i],
					  lmbd=lmbds[j],
					  epochs=1000,
					  softmax=True)
		

		logreg.train(Xtest, ytest)
		tsnn.train(Xtest, ytest)
		accuracies_our[i, j] = np.max(logreg.accuracies)
		accuracies_tsnn[i, j] = np.max(tsnn.accuracies)

p.plot_heatmap(accuracies_our, etas, lmbds, f'mse_heatmap_Cancer_sigmoid_log_regression.pdf')
p.plot_heatmap(accuracies_tsnn, etas, lmbds, f'mse_heatmap_Cancer_sigmoid_log_regression_scikit.pdf')
