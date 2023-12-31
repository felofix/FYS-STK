from functionality import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
from imageio import imread

# Introducing constant randomness.
np.random.seed(14)

def regress_and_plot_OLS(datatype):
	"""
	Datatype. Real of fake?
	Only accepts 'Real' or 'Fake'.
	"""
	if datatype == 'generated':
		# Generate data 
		n = 20
		x = np.linspace(0, 1, n)
		y = np.linspace(0, 1, n)
		x, y = np.meshgrid(x,y)
		z = FrankeFunction(x, y).ravel()
		z+= 0.2*np.random.randn(z.size)

	if datatype == 'real':
		# Load the terrain
		terrain1 = imread('terrain/SRTM_data_Norway_1.tif')[::100, ::100]
		xlen, ylen = terrain1.shape[1], terrain1.shape[0]
		x, y = np.arange(0, xlen), np.arange(0, ylen)
		x, y = np.meshgrid(x, y)
		z = terrain1
		z = (z - np.mean(z)) / np.std(z)

	# Make life easier, flat x and y 
	x_flat = x.flatten()
	y_flat = y.flatten()
	z_flat = z.flatten()

	# Creating design matrix. 
	nr_of_degrees = 10
	R2_scores_tr = np.zeros(nr_of_degrees)
	MSE_scores_tr = np.zeros(nr_of_degrees)
	betas_tr = []
	R2_scores_test = np.zeros(nr_of_degrees)
	MSE_scores_test = np.zeros(nr_of_degrees)
	betas_test = []

	for degree in range(1, nr_of_degrees+1):
		reg = Franke_Regression()
		X = reg.create_design_matrix(x_flat, y_flat, degree)
		X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.33, random_state=42)
		X_scaled_train, X_scaled_test = reg.scale(X_train, X_test) # Scaled.

		beta = reg.find_betas_OLS(X_scaled_train, z_train)
		z_pred_training = reg.predict_z(X_scaled_train, beta) + z_train.mean()
		z_pred_test = reg.predict_z(X_scaled_test, beta) + z_train.mean()

		# R2 and MSE.
		R2_scores_tr[degree - 1] = reg.R2_score(z_train, z_pred_training)
		MSE_scores_tr[degree - 1] = reg.MSE(z_train, z_pred_training)

		R2_scores_test[degree - 1] = reg.R2_score(z_test, z_pred_test)
		MSE_scores_test[degree - 1] = reg.MSE(z_test, z_pred_test)

		# Betas
		betas = reg.find_betas_OLS(X_scaled_train, z_train)
		betas_tr.append(betas)
		z_pred = reg.predict_z(X_scaled_train, betas)

		betas = reg.find_betas_OLS(X_scaled_test, z_test)
		betas_test.append(betas)
		z_pred = reg.predict_z(X_scaled_test, betas)

	degrees = np.arange(1,nr_of_degrees+1, 1)

	# Plotting
	plt.figure(figsize=(7, 5))
	plt.plot(degrees, MSE_scores_tr, marker='o', label='Training MSE')
	plt.plot(degrees, MSE_scores_test, marker='x', label='Test MSE')
	plt.xlabel('Polynomial Degree')
	plt.ylabel('MSE')
	plt.title(f'Mean Squared Error vs. Polynomial Degree with {datatype} data using OLS')
	plt.legend()
	plt.grid(True)
	plt.savefig(f"plots/MSE_OLS_{datatype}.pdf")

	# Plotting
	plt.figure(figsize=(7, 5))
	plt.plot(degrees, R2_scores_tr, marker='o', label='Training R2')
	plt.plot(degrees, R2_scores_test, marker='x', label='Test R2')
	plt.xlabel('Polynomial Degree')
	plt.ylabel('R2')
	plt.title(f'R2 vs. Polynomial Degree with {datatype} data using OLS' )
	plt.legend()
	plt.grid(True)
	plt.savefig(f"plots/R2_OLS_{datatype}.pdf")

	
	fig, ax = plt.subplots(figsize=(7, 5))
	colors = ['blue', 'green', 'red', 'purple', 'black']

	for degree, betas_degree in enumerate(betas_tr[:5], 1):
		color = colors[(degree-1) % len(colors)]  
		x_vals = np.arange(len(betas_degree))
		ax.scatter(x_vals, betas_degree, marker='o', color=color, label=f'Degree {degree}')
		ax.vlines(x_vals, 0, betas_degree, color=color, linestyle='--', lw=1)

	ax.set_xlabel('$\\beta_j$')
	ax.set_ylabel('Coefficient Value')
	ax.set_title(f'Beta Coefficients for {datatype} data')
	ax.legend()
	ax.grid(True)

	plt.tight_layout()
	plt.xticks(np.arange(0, 21, step=1))
	plt.savefig(f"plots/beta_coefficents__{datatype}.pdf")
	
if __name__ == "__main__":
	regress_and_plot_OLS('real')
	regress_and_plot_OLS('generated')


