import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('terrain/SRTM_data_Norway_1.tif')[::100, ::100]
# Show the terrain
xlen, ylen = terrain1.shape[1], terrain1.shape[0]
x, y = np.arange(0, xlen), np.arange(0, ylen)
xij, yij = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot a surface
ax.plot_surface(xij, yij, terrain1, cmap='viridis')

# Add labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Terrain model, real data')

# Show the plot
plt.show()



"""
plt.figure()
plt.title('Terrain over Norway 1')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""