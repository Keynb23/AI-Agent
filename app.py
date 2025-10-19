# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from perceptron_core import Perceptron 
from CNN import conceptual_classifier, CLASSES, IMAGE_FEATURES

app = Flask(__name__)
CORS(app) 

# ------------------------------------------------------------------
# Perceptron Code (Unchanged)
# ------------------------------------------------------------------
perceptron_model = None
perceptron_params = {"weights": [0, 0], "bias": 0}
AND_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
AND_Y = np.array([0, 0, 0, 1]) 

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "This was created to help me understand full-stack development and machine learning better!"})

@app.route('/api/perceptron/train', methods=['GET'])
def train_perceptron_route():
    train_perceptron() 
    return jsonify({
        "status": "Trained successfully!",
        "concept": "AND Gate Logic (X1 AND X2)",
        "weights": perceptron_params["weights"],
        "bias": perceptron_params["bias"],
        "explanation": "The model learned to output 1 only when both inputs are 1."
    })

def train_perceptron():
    global perceptron_model, perceptron_params
    perceptron_model = Perceptron(learning_rate=0.1, n_iters=10)
    perceptron_model.fit(AND_X, AND_Y)
    perceptron_params = {
        "weights": perceptron_model.weights.tolist(),
        "bias": float(perceptron_model.bias)
    }

@app.route('/api/perceptron/predict', methods=['POST'])
def predict_perceptron():
    data = request.get_json()
    input1 = data.get('input1', 0)
    input2 = data.get('input2', 0)
    w1, w2 = perceptron_params["weights"]
    b = perceptron_params["bias"]
    linear_output = (w1 * input1) + (w2 * input2) + b
    prediction = 1 if linear_output >= 0 else 0
    return jsonify({
        "input1": input1,
        "input2": input2,
        "weighted_sum": float(linear_output),
        "prediction": prediction,
        "weights_used": perceptron_params["weights"],
        "bias_used": perceptron_params["bias"]
    })

# ------------------------------------------------------------------
# CNN Routes (Modified)
# ------------------------------------------------------------------

@app.route('/api/cnn/predict', methods=['POST'])
def cnn_predict():
    data = request.get_json()
    image_id = data.get('imageId', 'default')
    # NEW: Get the list of classes to ignore from the frontend
    ignore_classes = data.get('ignore_classes', [])
    
    # Pass this list to the model
    prediction, scores = conceptual_classifier.predict(image_id, ignore_classes)
    
    return jsonify({
        "imageId": image_id,
        "prediction": prediction,
        "scores": scores,
        "model_weights": conceptual_classifier.weights.tolist(),
        "features": IMAGE_FEATURES.get(image_id, np.zeros(4)).tolist()
    })

@app.route('/api/cnn/feedback', methods=['POST'])
def cnn_feedback():
    data = request.get_json()
    image_id = data.get('imageId')
    # NEW: Get the one correct class
    correct_class = data.get('correct_class')
    # NEW: Get the list of incorrect guesses
    incorrect_classes = data.get('incorrect_classes', [])

    if not correct_class:
         return jsonify({"updated": False, "message": "No correct class provided."}), 400

    weights_updated = conceptual_classifier.update_weights(
        image_id, 
        correct_class, 
        incorrect_classes
    )

    return jsonify({
        "updated": weights_updated,
        "new_weights": conceptual_classifier.weights.tolist(),
        "message": "Weights adjusted." if weights_updated else "No adjustment needed."
    })

# ------------------------------------------------------------------
# App Start
# ------------------------------------------------------------------
if __name__ == '__main__':
    train_perceptron()
    app.run(debug=True, port=5000)