from functionality import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#
np.random.seed(14)

# Generate data 
n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y) + np.random.normal(scale = 0.01, size = (n,n))

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()

# Creating design matrix. 
nr_of_degrees = 20
R2_scores_tr = np.zeros(nr_of_degrees)
MSE_scores_tr = np.zeros(nr_of_degrees)
betas_tr = []
R2_scores_test = np.zeros(nr_of_degrees)
MSE_scores_test = np.zeros(nr_of_degrees)
betas_test = []

for degree in range(1, nr_of_degrees+1):
	reg = Franke_Regression()
	X = np.vstack((x_flat, y_flat)).T

	# Create a PolynomialFeatures object with degree=5
	poly = PolynomialFeatures(degree=degree)

	# Transform the data to polynomial features
	X = poly.fit_transform(X)
	X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)
	X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled. 
	
	regger = LinearRegression().fit(X_scaled_train, z_train)
	z_pred_train = regger.predict(X_scaled_train)
	MSE_scores_tr[degree - 1] = reg.MSE(z_train, z_pred_train)
	R2_scores_tr[degree - 1] = reg.R2_score(z_train, z_pred_train)

	z_pred_test = regger.predict(X_scaled_test)
	MSE_scores_test[degree - 1] = reg.MSE(z_test, z_pred_test)
	R2_scores_test[degree - 1] = reg.R2_score(z_test, z_pred_test)


	"""
	reg = Franke_Regression()
	X = reg.create_design_matrix(x_flat, y_flat, degree)

	# Train
	betas = reg.find_betas_OLS(X_train, z_train)
	betas_tr.append(betas)
	z_pred = reg.predict_z(X_train, betas)

	R2_scores_tr[degree - 1] = reg.R2_score(z_train, z_pred)
	MSE_scores_tr[degree - 1] = reg.MSE(z_train, z_pred)
	
	# Test
	betas = reg.find_betas_OLS(X_test, z_test)
	betas_test.append(betas)
	z_pred = reg.predict_z(X_test, betas)

	R2_scores_test[degree - 1] = reg.R2_score(z_test, z_pred)
	MSE_scores_test[degree - 1] = reg.MSE(z_test, z_pred)
	"""
	
	

all_degrees = np.arange(1,nr_of_degrees+1, 1)

plt.plot(all_degrees, MSE_scores_tr, 'o-b', label = "MSE Train")
plt.title("MSE as a function of degrees")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid()
plt.legend()
#plt.savefig("plots/MSEtrainingOLS.pdf")


plt.plot(all_degrees, MSE_scores_test, 'o-r', label = "MSE Test")
plt.title("MSE as a function of degrees")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid()
plt.legend()
plt.show()

plt.plot(all_degrees, R2_scores_tr, 'o-b', label = "R2 Train" )
plt.title("$R^2$ as a function of degrees")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid()
plt.legend()
#plt.savefig("plots/R2trainOLS.pdf")

plt.plot(all_degrees, R2_scores_test, 'o-r', label = "R2 Test" )
plt.title("$R^2$ as a function of degrees")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.grid()
plt.legend()
#plt.savefig("plots/R2testOLS.pdf")
plt.show()

"""
fig, ax = plt.subplots(figsize=(10, 6))

# Define a color map to have different colors for different degrees
colors = ['blue', 'green', 'red', 'purple', 'black']

for degree, betas_degree in enumerate(betas_tr, 1):
    x_vals = np.arange(len(betas_degree))
    ax.scatter(x_vals, betas_degree, marker='o', color=colors[degree-1], label=f'Degree {degree}')
    ax.vlines(x_vals, 0, betas_degree, color=colors[degree-1], linestyle='--', lw=1)  

ax.set_xlabel('$\\beta_j$')
ax.set_ylabel('Coefficient Value')
ax.set_title('Beta Coefficients')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.xticks(np.arange(0, 21, step=1))
plt.savefig("plots/beta_coefficents.pdf")
"""
