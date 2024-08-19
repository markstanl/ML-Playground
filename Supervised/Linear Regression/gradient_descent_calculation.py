def get_gradient_at_b(x, y, b, m) -> float:
  N = len(x)
  diff = 0
  for i in range(N):
    x_val = x[i]
    y_val = y[i]
    diff += (y_val - ((m * x_val) + b))
  b_gradient = -(2/N) * diff
  return b_gradient

# The gradient of m
def get_gradient_at_m(x, y, b, m)-> float:
  N = len(x)
  diff = 0
  for i in range(N):
      x_val = x[i]
      y_val = y[i]
      diff += x_val * (y_val - ((m * x_val) + b))
  m_gradient = -(2/N) * diff
  return m_gradient

# The step gradient function
def step_gradient(x: list, y: list, b_current: float, m_current: float, learning_rate: float = 0.01):
  b_gradient = get_gradient_at_b(x, y, b_current, m_current)
  m_gradient = get_gradient_at_m(x, y, b_current, m_current)
  # Simply get the gradients at the points

  b = b_current - (learning_rate * b_gradient)
  m = m_current - (learning_rate * m_gradient)
  # and apply them. The gradient tells us what direction will be of greatest increase, so we go the opposite
  return [b, m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 156, 184]

# Function to calculate average loss for visualization that the algorithm is working
def calculate_loss(x: list, y: list, b_current: float, m_current: float) -> float:
    total_loss = 0
    n = len(x)  # Number of data points
    for i in range(n):
        # Calculate the predicted value
        y_pred = m_current * x[i] + b_current
        # Calculate the squared error
        i_loss = 0.5 * (y_pred - y[i]) ** 2
        # Sum up the squared errors
        total_loss += i_loss
    # Calculate the mean squared error
    mean_loss = total_loss / n
    return mean_loss

b = 0
m = 0

# running this algorithm works
b, m = step_gradient(months, revenue, b, m)
print(b, m)
print(calculate_loss(months, revenue, b, m))