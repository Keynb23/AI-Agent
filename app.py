from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sys

# --- Conditional Import for Perceptron and Data ---
# This ensures the app doesn't crash if perceptron.py is missing definitions
try:
    # Assuming Perceptron class, X_or, and y_or are defined in perceptron.py
    from perceptron import Perceptron
    
    # Data Definition (if not in perceptron.py, define it here)
    X_or = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_or = np.array([0, 1, 1, 1])
    
except ImportError:
    print("FATAL ERROR: Could not import Perceptron or required data from perceptron.py.")
    print("Please ensure perceptron.py exists and contains the Perceptron class.")
    sys.exit(1)

# --- Flask App Initialization ---
# We explicitly set static_folder='static' for robustness
app = Flask(__name__, static_folder='static')

# 1. Initialize and Train Model on Startup
try:
    ppn_agent = Perceptron(learning_rate=0.5, epochs=10)
    ppn_agent.fit(X_or, y_or) 
    
    # Check if the model converged successfully (misclassifications is 0)
    if ppn_agent.errors_[-1] == 0:
        print("Agent Trained Successfully (OR Logic). Converged.")
    else:
        print(f"Agent Trained (OR Logic). Did NOT Converge. Final Error: {ppn_agent.errors_[-1]}")
        
    print(f"Final Weights: {ppn_agent.weights}, Final Bias: {ppn_agent.bias}")

except Exception as e:
    print(f"FATAL ERROR during Perceptron training: {e}")
    sys.exit(1)


@app.route('/')
def index():
    """Renders the main HTML interface from the templates folder."""
    # This route is the cause of the 'directory listing' error if Flask isn't running.
    # It must be the only function that calls render_template.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handles the prediction API request from app.js."""
    try:
        data = request.json
        x1 = int(data.get('x1', 0))
        x2 = int(data.get('x2', 0))

        # Input validation for binary inputs
        if x1 not in [0, 1] or x2 not in [0, 1]:
            return jsonify({'error': 'Inputs must be 0 or 1.'}), 400

        # Convert input to NumPy array
        input_data = np.array([[x1, x2]])

        # Get prediction
        prediction = ppn_agent.predict(input_data)[0]

        return jsonify({
            'x1': x1,
            'x2': x2,
            'prediction': int(prediction) # Ensure integer type for JSON
        })
    except Exception as e:
        # Catch and log any errors during prediction
        print(f"Prediction API Error: {e}")
        return jsonify({'error': 'Internal server error during prediction.'}), 500

if __name__ == '__main__':
    # --- Critical File Structure Check ---
    if not os.path.isdir('templates'):
        print("ERROR: 'templates' folder not found. Create it and place index.html inside.")
        sys.exit(1)
    
    if not os.path.isdir('static/css') or not os.path.isdir('static/js'):
        print("ERROR: 'static/css' or 'static/js' folders not found.")
        print("Ensure you have a 'static' folder with 'css' and 'js' subfolders.")
        sys.exit(1)
        
    # --- Run the Application ---
    # Using app.run() which is equivalent to 'flask run' but allows direct execution
    app.run(debug=True)