from functionality import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Same results every time. 
np.random.seed(14)

# Generate data 
n = 100
x = np.sort(np.random.uniform(0, 1, n))
y = np.sort(np.random.uniform(0, 1, n))
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) + 0.1 * np.random.normal(n,n)

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Creating design matrix. 
nr_of_degrees = 10
lambdas = [0.0001, 0.001, 0.01, 0.1, 1]

# Training scores.
R2_scores_tr = np.zeros((nr_of_degrees, len(lambdas)))
MSE_scores_tr = np.zeros((nr_of_degrees, len(lambdas)))

# Test scores. 
R2_scores_te = np.zeros((nr_of_degrees, len(lambdas)))
MSE_scores_te = np.zeros((nr_of_degrees, len(lambdas)))

x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_flat, y_flat, z_flat, test_size=0.2)

for l in range(len(lambdas)):
	for degree in range(1, nr_of_degrees+1):
		reg = Franke_Regression()
		X_train = reg.create_desgin_matrix(x_train, y_train, degree)
		X_scaled_train = reg.mean_scale(X_train)

		# Training.
		betas = reg.find_betas_Ridge(X_scaled_train, z_train, lambdas[l])
		z_pred = reg.predict_z(X_scaled_train, betas)
		R2_scores_tr[degree - 1, l] = reg.R2_score(z_train, z_pred)
		MSE_scores_tr[degree - 1, l] = reg.MSE(z_train, z_pred)

		# Test. 
		X_test = reg.create_desgin_matrix(x_test, y_test, degree)
		X_scaled_test = reg.test_scale(X_test) # Scaled with the same numbers. 
		z_pred_test = reg.predict_z(X_scaled_test, betas)
		R2_scores_te[degree - 1, l] = reg.R2_score(z_test, z_pred_test)
		MSE_scores_te[degree - 1, l] = reg.MSE(z_test, z_pred_test)


all_degrees = np.arange(1,nr_of_degrees+1, 1)

# Plotting training set. 

for l in range(len(lambdas)):
	plt.plot(all_degrees, MSE_scores_tr[:, l], label = f"lambda = {lambdas[l]}")

plt.legend()
plt.grid()
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.savefig("plots/MSEtrainingRidge.pdf")
plt.show()

for l in range(len(lambdas)):
	plt.plot(all_degrees, R2_scores_tr[:, l], label = f"lambda = {lambdas[l]}")

plt.legend()
plt.xlabel("Degree")
plt.ylabel("R2")
plt.grid()
plt.savefig("plots/R2trainingRidge.pdf")
plt.show()

# Plotting test set. 
for l in range(len(lambdas)):
	plt.plot(all_degrees, MSE_scores_te[:, l], label = f"lambda = {lambdas[l]}")

plt.legend()
plt.grid()
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.savefig("plots/MSEtestRidge.pdf")
plt.show()

for l in range(len(lambdas)):
	plt.plot(all_degrees, R2_scores_te[:, l], label = f"lambda = {lambdas[l]}")

plt.legend()
plt.xlabel("Degree")
plt.ylabel("R2")
plt.grid()
plt.savefig("plots/R2testRidge.pdf")
plt.show()
