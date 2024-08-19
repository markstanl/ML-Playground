import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D data
np.random.seed(42)
X = np.dot(np.random.rand(3, 3), np.random.randn(3, 200)).T

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the original data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.2)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

# Plot the principal components
ax.quiver(0, 0, 0, pca.components_[0, 0], pca.components_[0, 1], pca.components_[0, 2], color='red')
ax.quiver(0, 0, 0, pca.components_[1, 0], pca.components_[1, 1], pca.components_[1, 2], color='green')

plt.title('PCA in 3D')
plt.show()
