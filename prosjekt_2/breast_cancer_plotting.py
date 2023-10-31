import numpy as np 
import ffnn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import plotting as p
import activations as act

# MSE vs epoch to find optimal epoch.
ffnn = nn.FFNN(X,
			   y, 
			   epochs=100, 
			   n_outputs=1, 
			   n_hidden_layers=2, 
			   n_hidden_neurons = 5,
			   softmax=True)
