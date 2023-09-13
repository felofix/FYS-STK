from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Oppgave a)

# Generate data 

n = 100
x = np.sort(np.random.random(n))
y = np.sort(np.random.random(n))
x, y = np.meshgrid(x,y)

# Frankie funciton + noise
z = FrankeFunction(x, y) + 0.1* np.random.normal(n,n)

# Make life easier, flat x and y 

x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()


def create_desgin_matrix(x,y, order):
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

X = create_desgin_matrix(x_flat,y_flat,5)

def find_betas(X,z):
    return np.linalg.inv(X.T@X)@X.T@z

def predict_z(X, betas):
    return X@betas

def MSE(z, z_pred):
    return 1/len(z)*np.sum( (z-z_pred)**2 )

def R2_score(z, z_pred):
    z_mean = np.mean(z)
    return 1 - np.sum( (z-z_pred)**2 ) / np.sum( (z-z_mean)**2 ) 

nr_of_degrees = 5
R2_scores = np.zeros(nr_of_degrees)
MSE_scores = np.zeros(nr_of_degrees)

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_flat, y_flat, z_flat, test_size=0.2)


for degree in range(1, nr_of_degrees+1):
    X_train = create_desgin_matrix(x_train, y_train, degree)
    X_test = create_desgin_matrix(x_test, y_test, degree)

    betas = find_betas(X_train, z_train)
    z_pred = predict_z(X_test, betas)

    R2_scores[degree - 1] = R2_score(z_test, z_pred)
    MSE_scores[degree - 1] = MSE(z_test, z_pred)


all_degrees = np.arange(1,nr_of_degrees+1, 1)

plt.plot(all_degrees, MSE_scores, label = "MSE")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()

plt.plot(all_degrees, R2_scores, label = "R2")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()

