import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# GENERATED WITH CHATGPT FOR VISUALIZATION

# Step 1: Generate synthetic training data
np.random.seed(0)
X_train = np.linspace(-10, 10, 100).reshape(-1, 1)
y_train = (X_train > 0).astype(int).ravel()
y_train = np.random.binomial(1, 0.8, size=len(y_train)) * y_train

# Plot the training data
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Training Data')
plt.legend()
plt.show()

# Step 2: Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 3: Generate test data
X_test = np.linspace(-10, 10, 200).reshape(-1, 1)

# Step 4: Predict probabilities on the test data
y_prob = model.predict_proba(X_test)[:, 1]

# Step 5: Plot the fitted sigmoid function
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.plot(X_test, y_prob, 'r', label='Sigmoid function')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision boundary (0.5)')
plt.axhline(0, color='gray', linestyle='--', xmin=-10, xmax=10, label='y=0')
plt.axhline(1, color='gray', linestyle='--', xmin=-10, xmax=10, label='y=1')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.show()
