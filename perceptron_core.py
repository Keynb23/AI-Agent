# perceptron_core.py
import numpy as np 

class Perceptron:
    """
    The Perceptron is the single simplest form of a neural network.
    It learns a linear boundary (like a line on a graph) to separate two types of data.
    
    Analogy: A simple decision-maker.
    """
    def _step_function(self, x):
        """Activation Function: Decides 1 (YES) or 0 (NO)."""
        # If the weighted sum (x) is greater than or equal to 0, output 1 (YES), otherwise 0 (NO).
        return np.where(x >= 0, 1, 0)
    
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._step_function
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """Trains the model to find the best weights and bias."""
        n_samples, n_features = X.shape
        # Initializing weights randomly (small values close to 0)
        self.weights = np.random.uniform(low=-0.01, high=0.01, size=n_features) 
        self.bias = np.random.uniform(low=-0.01, high=0.01) # Initializing bias randomly
        
        # Ensure target labels are 0 or 1
        y_ = np.where(y > 0, 1, 0)
        
        for _ in range(self.n_iters):
            for x_i, target in zip(X, y_):
                
                # 1. Calculate Weighted Sum (Linear Output)
                # Formula: (W1*X1) + (W2*X2) + Bias
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # 2. Get Prediction (0 or 1)
                y_predicted = self.activation_func(linear_output)
                
                # 3. Calculate Error and Update (Only if prediction is wrong)
                error = target - y_predicted # Error: (Correct Answer - Prediction)
                
                # Update Rule: If error is not 0, adjust W and B
                if error != 0:
                    update = self.lr * error
                    self.weights += update * x_i # Adjust weights
                    self.bias += update          # Adjust bias

    def predict(self, X):
        """Predicts class labels (0 or 1) for new data."""
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
