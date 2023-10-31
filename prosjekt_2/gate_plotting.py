import numpy as np
import ffnn as nn
import plotting as p

# Random seed. 
np.random.seed(0)

# Design matrix.
X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)

# Outputs. 
xor_gate = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
and_gate = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
or_gate  = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
gates = {'xor_gate': xor_gate, 'and_gate': and_gate, 'or_gate': or_gate}

etas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
lmbds = [1e-4, 1e-3, 1e-2, 1e-1, 1]

for gate in gates:
	accuracies = np.zeros((len(etas), len(lmbds)))

	for i in range(len(etas)):
		for j in range(len(lmbds)):
			ffnn = nn.FFNN(X, gates[gate], epochs=1000, eta=etas[i], lmbd=lmbds[j], n_hidden_layers=1, n_hidden_neurons = 2, softmax=True)
			ffnn.train(X, gates[gate])
			accuracies[i, j] = np.max(ffnn.accuracies)

	p.plot_heatmap(accuracies, etas, lmbds, f'mse_heatmap_{gate}.pdf')