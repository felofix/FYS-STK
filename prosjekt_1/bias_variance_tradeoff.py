import numpy as np
import matplotlib.pyplot as plt 
from functionality import *
from sklearn.utils import resample

#
np.random.seed(2018)

# Generate data 
n = 40
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y).ravel()
z+= 0.1*np.random.randn(z.size)

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_flat, y_flat, z_flat, test_size=0.2)

# Define some constants before doing the bias variance trade-off analysis

n_boostraps = 100
maxdegree = 20

error = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)

for degree in range(1, maxdegree+1):
    reg = Franke_Regression()
    z_pred = np.empty((len(z_test), n_boostraps))

    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        X_train = reg.create_design_matrix(x_, y_, degree)
        betas = reg.find_betas_OLS(X_train, z_)
        
        X_test = reg.create_design_matrix(x_test, y_test, degree)
        z_pred[:, i] = reg.predict_z(X_test, betas).ravel()

    polydegree[degree-1] = degree
    error[degree-1] = np.mean(np.mean((z_test[:, np.newaxis] - z_pred)**2, axis=1, keepdims=True))
    bias[degree-1] = np.mean( (z_test[:, np.newaxis] - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree-1] = np.mean( np.var(z_pred, axis=1, keepdims=True) )

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.legend()
plt.grid()
plt.savefig("plots/bias_variance_tradeoff.pdf")
plt.show()