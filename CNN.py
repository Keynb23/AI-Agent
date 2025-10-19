# CNN.py
import numpy as np

CLASSES = ["Cat", "Dog", "Person"]
CLASS_MAP = {cls: i for i, cls in enumerate(CLASSES)}

IMAGE_FEATURES = {
    "cat_image": np.array([0.8, 0.2, 0.1, 0.5]), 
    "dog_image": np.array([0.1, 0.7, 0.3, 0.9]),
    "person_image": np.array([0.3, 0.4, 0.9, 0.2]),
}

class ConceptualClassifier:
    def __init__(self, n_features=4, n_classes=3, learning_rate=0.1):
        self.lr = learning_rate
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(n_features, n_classes))
        
    def predict(self, image_id, ignore_classes=[]):
        features = IMAGE_FEATURES.get(image_id)
        if features is None:
            features = np.zeros(self.n_features)

        scores = np.dot(features, self.weights)
        
        for class_name in ignore_classes:
            if class_name in CLASS_MAP:
                ignore_index = CLASS_MAP[class_name]
                scores[ignore_index] = -np.inf # Set to -infinity

        predicted_index = np.argmax(scores)
        predicted_class = CLASSES[predicted_index]
        
        # --- *** THIS IS THE FIX *** ---
        # We must convert 'scores' to a list that is JSON-safe.
        # np.nan_to_num converts -np.inf to a very large negative number (e.g., -1e10)
        # which JSON *can* handle.
        scores_finite = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Return the JSON-safe list
        return predicted_class, scores_finite.tolist()

    def update_weights(self, image_id, correct_class, incorrect_classes):
        features = IMAGE_FEATURES.get(image_id)
        if features is None:
            return False

        if correct_class in CLASS_MAP:
            true_index = CLASS_MAP[correct_class]
            self.weights[:, true_index] += self.lr * features
        
        for class_name in incorrect_classes:
            if class_name in CLASS_MAP:
                predicted_index = CLASS_MAP[class_name]
                self.weights[:, predicted_index] -= self.lr * features
        
        return True

conceptual_classifier = ConceptualClassifier()