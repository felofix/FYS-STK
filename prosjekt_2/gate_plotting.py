import numpy as np
import ffnn as nn
import plotting as p
import activations as act

# Random seed. 
np.random.seed(0)

# Design matrix.
X = np.array([[0, 0], [0, 1], [1, 0],[1, 1]], dtype=np.float64)

# Outputs. 
xor_gate = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
and_gate = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
or_gate  = np.array([[1, 0], [0, 1], [0, 1], [0, 1]])
gates = {'xor_gate': xor_gate, 'and_gate': and_gate, 'or_gate': or_gate}
activations = {'sigmoid': act.sigmoid, 'relu': act.RELU,  'lrelu': act.LRELU}

etas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
lmbds = [1e-4, 1e-3, 1e-2, 1e-1, 1]

for activation in activations:
	for gate in gates:
		accuracies = np.zeros((len(etas), len(lmbds)))

		for i in range(len(etas)):
			for j in range(len(lmbds)):
				ffnn = nn.FFNN(X, gates[gate], epochs=1000,
							   eta=etas[i], lmbd=lmbds[j], n_hidden_layers=1, 
							   n_hidden_neurons = 2, softmax=True, batch_size=1, 
							   activation_function=activations[activation], verbose=False)
				ffnn.train(X, gates[gate])
				accuracies[i, j] = np.max(ffnn.accuracies)

		if activation == 'sigmoid' and gate == 'xor_gate':
			p.plot_heatmap(accuracies, etas, lmbds, 'accuracy_heatmap_sigmoid_xor_gate.pdf')

		print(f'Best acc: {np.nanmax(accuracies*100):.2f}')
		print(f'Activation function: {activation}')
		print(f'Gate: {gate}')