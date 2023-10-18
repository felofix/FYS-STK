import numpy as np 
import neural_network as nn

def create_design_matrix(x, order):
	""" Only for testing. 
	"""
	order += 1

	X = np.zeros((len(x), order))

	for i in range(order):
		X[:,i] = x**(i)
		
	return X

x = np.linspace(0, 10, 100)
y = x*x + 2*x 
X = create_design_matrix(x, 2)
FNNN = nn.FNNN(X, y)
FNNN.train()




""" # Plotting. 
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
"""