from functionality import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Generate data 
n = 100
x = np.sort(np.random.random(n))
y = np.sort(np.random.random(n))
x, y = np.meshgrid(x,y)
z = (FrankeFunction(x, y)+ 0.1* np.random.normal(n,n)).flatten()

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Creating design matrix. 
nr_of_degrees = 10
lambdas = [0.0001, 0.001, 0.01, 0.1, 1]

R2_scores = np.zeros((nr_of_degrees, len(lambdas)))
MSE_scores = np.zeros((nr_of_degrees, len(lambdas)))

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_flat, y_flat, z_flat, test_size=0.2)

for l in range(len(lambdas)):
	for degree in range(1, nr_of_degrees+1):
		reg = Franke_Regression()
		X_train = reg.create_desgin_matrix(x_train, y_train, degree)
		X_scaled = reg.mean_scale(X_train)
		betas = reg.find_betas_Ridge(X_scaled, z_train, lambdas[l])
		z_pred = reg.predict_z(X_scaled, betas)
		R2_scores[degree - 1, l] = reg.R2_score(z_train, z_pred)
		MSE_scores[degree - 1, l] = reg.MSE(z_train, z_pred)


all_degrees = np.arange(1,nr_of_degrees+1, 1)
plt.plot(all_degrees, MSE_scores, label = "MSE")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()

plt.plot(all_degrees, R2_scores, label = "R2")
plt.xlabel("Degree")
plt.ylabel("R2")
plt.show()
