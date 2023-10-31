import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_mse_vs_epochs(epochs, mses, title):
	# Set the style of the visualization
	sns.set_theme(style="whitegrid")

	# Create a color palette
	palette = sns.color_palette("husl", 1)

	# Create a line plot of 'mses' vs 'epochs'
	plt.figure(figsize=(10, 6))
	sns.lineplot(x=epochs, y=mses, palette=palette, linewidth=2.5)

	msemin = np.min(mses)
	epochmin = np.argwhere(mses == msemin)

	plt.scatter(epochs[epochmin], msemin,color='orange', label=f'Minimum MSE: {msemin:.4f}')

	# Set title and labels for axes
	plt.legend()
	plt.title("MSE vs Epochs", fontsize=20)
	plt.xlabel("Epochs", fontsize=14)
	plt.ylabel("Mean Squared Error (MSE)", fontsize=14)

	plt.savefig(title)

def plot_heatmap(matrix, row_labels, col_labels, title, cmap="YlGnBu"):
	"""
	Plot a heatmap using Seaborn.

	Parameters:
		matrix (numpy.array): 2D array to be plotted.
		row_labels (list): Labels for the rows.
		col_labels (list): Labels for the columns.
		title (str): Title of the heatmap.
		cmap (str, optional): Color map. Defaults to "YlGnBu".

	Returns:
		None
	"""
	plt.figure(figsize=(10, 8))
	
	# Create a heatmap using Seaborn
	sns.heatmap(matrix, annot=True, cmap=cmap, cbar=True, 
				xticklabels=col_labels, yticklabels=row_labels, 
				fmt=".2e", linewidths=0.5)

	# Setting the title
	plt.title(title, fontsize=18)
	
	# Setting x and y labels
	plt.xlabel('Lambda Values', fontsize=14)
	plt.ylabel('Eta Values', fontsize=14)

	plt.savefig("plots/" + title)