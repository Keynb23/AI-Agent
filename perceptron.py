import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """A simple Perceptron classifier."""

    def __init__(self, learning_rate=0.5, epochs=10):
        # Hyperparameters
        self.lr = learning_rate # Controls the size of weight updates
        self.epochs = epochs    # Number of training passes over the dataset
        # Model Parameters (initialized when training starts)
        self.weights = None
        self.bias = None
        self.errors_ = [] # List to track misclassifications per epoch

    def _step_function(self, x):
        """The activation function (threshold logic)."""
        # Returns 1 if net input is 0 or greater, otherwise 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Train the Perceptron model."""
        # 1. Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features) 
        self.bias = 0
        
        for _ in range(self.epochs):
            n_errors = 0
            for xi, target in zip(X, y):
                # 2. Forward Propagation (Calculate the net input)
                # net_input = W . X + b
                net_input = np.dot(xi, self.weights) + self.bias
                
                # 3. Predict the output (y_hat)
                y_pred = self._step_function(net_input)
                
                # 4. Calculate the error (Error = Target - Prediction)
                error = target - y_pred
                
                # 5. Update weights and bias (The Learning Rule)
                if error != 0:
                    update = self.lr * error
                    # Update Rule: W_new = W_old + (lr * error * xi)
                    self.weights += update * xi 
                    
                    # --- BUG FIX #1 HERE ---
                    # CORRECT: self.bias += update 
                    self.bias += update         
                    
                # --- BUG FIX #2 HERE ---
                # CORRECT: n_errors += int(error != 0.0) 
                n_errors += int(error != 0.0)
            
            self.errors_.append(n_errors) # Store misclassifications for this epoch
            
            # OPTIONAL STOPPING CONDITION: If the error is zero, the model has converged
            if n_errors == 0:
                print(f"Converged at Epoch {_ + 1}")
                break

        return self

    def predict(self, X):
        """Generate the prediction for new data."""
        # Calculate the net input for all samples
        net_input = np.dot(X, self.weights) + self.bias
        
        # Apply the step function
        return self._step_function(net_input)
# --- TESTING THE MODEL WITH THE OR GATE ---

# Input Data for OR gate (Same inputs as AND)
X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target Labels for OR gate (Different from AND)
# OR Logic: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
y_or = np.array([0, 1, 1, 1])

# Initialize and train the Perceptron
ppn = Perceptron(learning_rate=0.5, epochs=10)
ppn.fit(X_or, y_or) # Make sure you pass the OR data!

# Make a prediction using the trained model
predictions = ppn.predict(X_or)

print('--- Perceptron Trained on OR Gate ---')
print(f'Model Predictions: {predictions}')
print(f"Final Weights: {ppn.weights}")
print(f"Final Bias: {ppn.bias}")

# --- TESTING THE MODEL WITH THE AND GATE ---

# Input Data for AND gate
# X_and = np.array([
#     [0, 0], # Output should be 0
#     [0, 1], # Output should be 0
#     [1, 0], # Output should be 0
#     [1, 1]  # Output should be 1
# ])

# Target Labels for AND gate
# y_and = np.array([0, 0, 0, 1])

# Initialize and train the Perceptron
# We'll stick with the correct learning_rate=0.5 now
# ppn = Perceptron(learning_rate=0.5, epochs=10) 
# ppn.fit(X_and, y_and)

# Make a prediction using the trained model
# predictions = ppn.predict(X_and)

# print('--- Perceptron Trained on AND Gate ---')
# print(f'Input Data (X):\n{X_and}')
# print(f"Target Labels (y): {y_and}")
# print(f"Model Predictions: {predictions}")
# print(f"Final Weights: {ppn.weights}")
# print(f"Final Bias: {ppn.bias}")

# Optional: Plot the error convergence
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassifications')
plt.title('Perceptron Error Convergence (Success)')
plt.show()
