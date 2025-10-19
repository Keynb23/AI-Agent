# perceptron.py
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    The Perceptron is the simplest type of artificial neural network.
    It learns a linear boundary to separate two classes of data.
    
    ☕ Analogy: The "brain" that decides if you can afford a $3.00 coffee.
    """

    def __init__(self, learning_rate=0.5, epochs=10):
        """Initializes weights, bias, and learning parameters."""
        self.lr = learning_rate 
        self.epochs = epochs    
        self.weights = None     
        self.bias = None        
        self.errors_ = []       

    def _step_function(self, x):
        """
        Activation Function: Translates the 'Confidence Score' into a final decision.
        :return: 1 (Afford/YES) if score >= 0, otherwise 0 (Can't Afford/NO).
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Trains the Perceptron model by adjusting weights and bias on errors."""
        n_features = X.shape[1]
        self.weights = np.zeros(n_features) # Start W1 ($5 Bill) and W2 (Change) at 0
        self.bias = 0                       # Start Bias at 0 (Initial Affordability Threshold)
        
        print(f"\n--- START TRAINING (Goal: Learn 'Afford Coffee' OR Logic) ---")
        
        for epoch in range(self.epochs):
            n_errors = 0
            
            # Print weights at the start of each learning round
            print(f"\n[EPOCH {epoch + 1}] | Initial Weights: {self.weights}, Bias: {self.bias:.2f}")
            
            for i, (xi, target) in enumerate(zip(X, y)):
                
                # 1. Forward Propagation: Calculate the total weighted input (Confidence Score)
                # Formula: Confidence = (W1 * X1) + (W2 * X2) + Bias
                net_input = np.dot(xi, self.weights) + self.bias
                y_pred = self._step_function(net_input)
                error = target - y_pred
                
                print(f"  Trial {i}: X={xi} ($5 Bill, Change), Target={target} (Correct Answer)")
                print(f"    Confidence: {net_input:.2f} -> Prediction: {y_pred}")
                
                # 2. Update Rule: Adjust weights and bias ONLY if there was an error
                if error != 0:
                    update = self.lr * error
                    
                    self.weights += update * xi 
                    self.bias += update
                    n_errors += 1
                    print(f"    *** MISTAKE! Error: {error}. New W/B: {self.weights}, {self.bias:.2f}")
                else:
                    print(f"    ✅ CORRECT! No update needed.")

            self.errors_.append(n_errors) 
            
            if n_errors == 0:
                print(f"\nConverged at Epoch {epoch + 1}. No errors remain!")
                break

        return self

    def predict(self, X):
        """Generates the final decision (1 or 0) for new data."""
        net_input = np.dot(X, self.weights) + self.bias
        return self._step_function(net_input)

# -------------------------------------------------------------
# ☕ TRAINING DATA: Affordability OR Gate Logic
# -------------------------------------------------------------

# X_or: [$5 Bill (X1), Change (X2)]
X_or = np.array([
    [0, 0], # No $5, No Change -> Target: 0 (Can't Afford)
    [0, 1], # No $5, Has Change -> Target: 1 (Afford)
    [1, 0], # Has $5, No Change -> Target: 1 (Afford)
    [1, 1]  # Has $5, Has Change -> Target: 1 (Afford)
])
y_or = np.array([0, 1, 1, 1])

# Initialize and train the Perceptron
ppn = Perceptron(learning_rate=0.5, epochs=10)
ppn.fit(X_or, y_or) 

# -------------------------------------------------------------
# --- CUSTOM TEST SECTION ---

def custom_test(model, x1_five_bill, x2_change):
    """
    Tests the trained Perceptron and prints the final YES/NO decision.
    
    Inputs:
    x1_five_bill: 1 (Has $5) or 0 (No $5)
    x2_change:    1 (Has $3 change) or 0 (No $3 change)
    """
    test_data = np.array([[x1_five_bill, x2_change]])
    weights = model.weights
    bias = model.bias
    
    # Calculate the Confidence Score
    confidence_score = np.dot(test_data[0], weights) + bias
    
    prediction = model.predict(test_data)[0]
    
    print("\n" + "="*50)
    print("--- FINAL AFFORDABILITY DECISION ---")
    
    # Translating the 1/0 prediction to YES/NO
    final_decision = "✅ YES, I have enough money (1)" if prediction == 1 else "❌ NO, I don't have enough money (0)"
    
    print(f"Scenario: $5 Bill={x1_five_bill}, Change={x2_change}")
    print(f"Trained Logic: ({weights[0]:.2f}*X1) + ({weights[1]:.2f}*X2) + {bias:.2f}")
    print(f"Confidence Score: {confidence_score:.2f}")
    print(f"FINAL DECISION: {final_decision}")
    print("="*50)


# ----------------------------------------------------------------------------------
# 🎯 YOUR TURN: CHANGE THESE NUMBERS (0 or 1) TO TEST SCENARIOS
# ----------------------------------------------------------------------------------

# ☕ SCENARIO 1: You have a $5 Bill (1), but no Change (0).
# Expected: YES, you can afford it.
FIVE_BILL_INPUT = 1 
CHANGE_INPUT = 0 
custom_test(ppn, x1_five_bill=FIVE_BILL_INPUT, x2_change=CHANGE_INPUT) 

# ☕ SCENARIO 2: You have NO $5 Bill (0), and NO Change (0).
# Expected: NO, you cannot afford it.
FIVE_BILL_INPUT = 0 
CHANGE_INPUT = 0 
custom_test(ppn, x1_five_bill=FIVE_BILL_INPUT, x2_change=CHANGE_INPUT)

# ☕ SCENARIO 3: You have NO $5 Bill (0), but you have Change (1).
# Expected: YES, you can afford it.
FIVE_BILL_INPUT = 0
CHANGE_INPUT = 1 
custom_test(ppn, x1_five_bill=FIVE_BILL_INPUT, x2_change=CHANGE_INPUT)

# ----------------------------------------------------------------------------------
# 📊 VISUALIZATION (THE GRAPH)
# ----------------------------------------------------------------------------------

# The graph shows the LEARNING RATE/PROGRESS.
# It tells you how long it took the Perceptron to get the correct weights and bias.
# You want to see the line quickly drop to 0, which means the model converged fast!

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

# X-Axis: Epochs (Each full training run through the 4 scenarios)
plt.xlabel('Epochs (Practice Rounds)')

# Y-Axis: Number of Misclassifications (How many of the 4 scenarios the model got WRONG in that round)
plt.ylabel('Number of Misclassifications (Wrong Decisions)') 

plt.title('Perceptron Learning Progress (Affordability)')
plt.show()