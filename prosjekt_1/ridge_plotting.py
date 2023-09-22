from functionality import *
from sklearn.linear_model import Ridge

# Same results every time. 
np.random.seed(14)

# Generate data 
n = 50
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y).ravel()
z+= 0.5*np.random.randn(z.size)

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Creating design matrix. 
nr_of_degrees = 10
lambdas = [1e-9, 1e-6, 1e-3, 1, 10]

# Training scores.
R2_scores_tr = np.zeros((nr_of_degrees, len(lambdas)))
MSE_scores_tr = np.zeros((nr_of_degrees, len(lambdas)))

# Test scores. 
R2_scores_te = np.zeros((nr_of_degrees, len(lambdas)))
MSE_scores_te = np.zeros((nr_of_degrees, len(lambdas)))

for l in range(len(lambdas)):
	for degree in range(1, nr_of_degrees+1):
		# Creating the model.
		reg = Franke_Regression()
		X = reg.create_design_matrix(x_flat, y_flat, degree)
		X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2, random_state=42)
		X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled. 
		
		# Training and prediciting.
		betas = reg.find_betas_Ridge(X_scaled_train, z_train, lambdas[l])
		z_pred_training = reg.predict_z(X_scaled_train, betas) + z_train.mean()
		z_pred_test = reg.predict_z(X_scaled_test, betas) + z_train.mean()

		# R2 and MSE.
		R2_scores_tr[degree - 1, l] = reg.R2_score(z_train, z_pred_training)
		MSE_scores_tr[degree - 1, l] = reg.MSE(z_train, z_pred_training)

		R2_scores_te[degree - 1, l] = reg.R2_score(z_test, z_pred_test)
		MSE_scores_te[degree - 1, l] =  reg.MSE(z_test, z_pred_test)


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

