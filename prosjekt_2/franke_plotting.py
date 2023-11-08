import numpy as np 
import ffnn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotting as p
import activations as act
import tensorffnn as ts

def FrankeFunction(x, y):
	term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
	term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
	term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))	
	term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
	return term1 + term2 + term3 + term4

def create_design_matrix(x, y, order):
	    if (len(x) != len(y)):
	        AssertionError("x and y must have the same length!")

	    number_of_combinations = int((order+1)*(order+2)/2)
	    X = np.zeros((len(x), number_of_combinations))

	    col_nr = 0

	    for i in range(order+1):
	        for j in range(i+1):
	            X[:,col_nr] = x**(i-j)*y**(j)
	            col_nr += 1
	    
	    return X


if __name__ == '__main__':	
	# Creating data.
	n = 100
	x = np.linspace(0, 1, n)
	y = np.linspace(0, 1, n)
	x, y = np.meshgrid(x,y)
	z = FrankeFunction(x, y).ravel()
	z+= 0.2*np.random.randn(z.size)			# Noise. 

	# Make life easier, flat x and y.
	x = x.flatten()
	y = y.flatten()
	z = z.flatten()

	X = create_design_matrix(x, y, 3)

	scaler = StandardScaler()
	scaler.fit(X)
	Xscaled = scaler.transform(X)
	X = Xscaled
	Xtrain, Xtest, ytrain, ytest = train_test_split(X, z, test_size=0.2)

	# Plotting optimaal epochs. 
	ffnn = nn.FFNN(Xtrain, ytrain, epochs=4000, n_outputs=1, eta=1e-5,lmbd=0.1, n_hidden_layers=2, n_hidden_neurons = 5)
	ffnn.train(Xtest, ytest)
	msesog = ffnn.mse 
	epochs = np.arange(len(msesog))*100

	p.plot_mse_vs_epochs(epochs, msesog, "epochs_mse_franke.pdf")

	# Plotting heatmaps. 
	etas = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
	lmbds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
	activation_functions = {'sigmoid': act.sigmoid}#, 'relu': act.RELU, 'lrelu': act.LRELU}

	for a in activation_functions:
		mses = np.zeros((len(etas), len(lmbds)))
		r2s = np.zeros((len(etas), len(lmbds)))

		for i in range(len(etas)):
			for j in range(len(lmbds)):
				ffnn = nn.FFNN(Xtrain, 
							   ytrain, 
							   epochs=500, 
							   n_outputs=1, 
							   eta=etas[i],
							   lmbd=lmbds[j], 
							   n_hidden_layers=2, 
							   n_hidden_neurons = 5,
							   activation_function=activation_functions[a])
				ffnn.train(Xtest, ytest)
				mses[i, j] = np.min(ffnn.mse)
				r2s[i, j] = np.min(ffnn.r2)

		p.plot_heatmap(mses, etas, lmbds, f'mse_heatmap_Franke_{a}.pdf')
		p.plot_heatmap(mses, etas, lmbds, f'mse_heatmap_Franke_{a}.pdf')
	
	# Plotting vs tensorflow. 
	tsnn = ts.FFNN(n_features = Xtrain.shape[1], n_outputs=1, n_hidden_layers=2, n_hidden_neurons=5, lmbd=0.1, eta=1e-5, epochs=4000)
	tsnn.train(Xtest, ytest)
	msests = tsnn.mses

	p.plot_mse_vs_tensorflow(epochs, msesog, msests, "our_ffnn_vs_tensorflow_franke.pdf")
	






