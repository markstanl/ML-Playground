# CHAT GPT GENERATED FOR VISUALIZATION

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate random 2D data
np.random.seed(42)
X = np.dot(np.random.rand(2, 2), np.random.randn(2, 200)).T

# Fit PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Plot the original data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.xlabel('X1')
plt.ylabel('X2')

# Plot the principal component
plt.plot([-2, 2], [-2*pca.components_[0, 1]/pca.components_[0, 0], 2*pca.components_[0, 1]/pca.components_[0, 0]], color='red')
plt.scatter(X_pca, np.zeros_like(X_pca), color='red')
plt.title('PCA in 2D')
plt.grid(True)
plt.show()
