import { useState, useEffect } from "react";
import "./App.css";

// --- API KEYS & URLS ---
const DOG_API_KEY = "live_2xty9ifc60ca60PWCtDVOJgq4DwNje7wts0CRWCRcHekoPljmRXeJTd9tR7dHzEs";
const DOG_API_URL = "https://api.thedogapi.com/v1/images/search?size=med&mime_types=jpg,png&limit=1";
const CAT_API_KEY = "live_p2kiswbNZhWuOxPsRwG834fOZnhGZ9l5nBns3boNrqB30ahpykeDq5gKryExJNs9";
const CAT_API_URL = "https://api.thecatapi.com/v1/images/search?size=med&mime_types=jpg,png&limit=1";
const PERSON_API_URL = "https://thispersondoesnotexist.com/";
const CLASSES = ["Cat", "Dog", "Person"];

// ------------------------------------------------------------------
// PerceptronModel Component
// ------------------------------------------------------------------
const PerceptronModel = () => {
  // (This component is unchanged)
  const [w1, setW1] = useState(null);
  const [w2, setW2] = useState(null);
  const [bias, setBias] = useState(null);
  const [status, setStatus] = useState("Loading...");
  const [input1, setInput1] = useState(1);
  const [input2, setInput2] = useState(1);
  const [prediction, setPrediction] = useState(null);
  const [weightedSum, setWeightedSum] = useState(null);
  const trainModel = async () => {
    try {
      const response = await fetch("http://localhost:5000/api/perceptron/train");
      const data = await response.json();
      setW1(data.weights[0].toFixed(2));
      setW2(data.weights[1].toFixed(2));
      setBias(data.bias.toFixed(2));
      setStatus(`Trained successfully on: ${data.concept}`);
    } catch (error) {
      setStatus("Error training model. Ensure Flask server is running on port 5000.");
      console.error("Training Error:", error);
    }
  };
  useEffect(() => {
    trainModel();
  }, []); 
  const handlePrediction = async () => {
    if (w1 === null) return; 
    try {
      const response = await fetch("http://localhost:5000/api/perceptron/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input1: Number(input1), input2: Number(input2) }),
      });
      const data = await response.json();
      setPrediction(data.prediction);
      setWeightedSum(data.weighted_sum.toFixed(4));
    } catch (error) {
      setPrediction("Error in prediction.");
      console.error("Prediction Error:", error);
    }
  };
  return (
    <div className="perceptron-model card">
      <h2>1. The Perceptron: The Single Neuron 🧠</h2>
      <p className="concept-explanation">
        The Perceptron is the *simplest neural network* (a single neuron) used for basic classification. It determines a decision (1 or 0) based on weighted inputs.
      </p>
      <div className="code-example">
        <h3>Conceptual Code: How a Neuron Decides</h3>
        <pre>
          <code>
            {`// The Decision Formula (Activation Function)
WeightedSum = (Input1 * W1) + (Input2 * W2) + Bias
// Step Function
Decision = 1 if WeightedSum >= 0 else 0 
// Example (AND Gate Logic):
// Output is 1 only if Input1 AND Input2 are 1.`}
          </code>
        </pre>
      </div>
      <div className="model-params">
        <p>Status: <strong>{status}</strong></p>
        <p>Learned Weights (Importance): W1 = **{w1 || '...'}**, W2 = **{w2 || '...'}**</p>
        <p>Learned Bias (Threshold): B = **{bias || '...'}**</p>
      </div>
      <hr style={{width: '90%', margin: '15px auto', borderTop: '1px solid var(--border-color)'}} />
      <div className="model-interaction">
        <h3>Test the Neuron: Predict AND({input1}, {input2})</h3>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center', margin: '10px 0', justifyContent: 'center'}}>
          <label>
            Input 1 (X1):
            <input 
              type="number" 
              min="0" 
              max="1" 
              value={input1} 
              onChange={(e) => setInput1(e.target.value)} 
              className="input-small"
            />
          </label>
          <label>
            Input 2 (X2):
            <input 
              type="number" 
              min="0" 
              max="1" 
              value={input2} 
              onChange={(e) => setInput2(e.target.value)} 
              className="input-small"
            />
          </label>
          <button className="button" onClick={handlePrediction}>Predict</button>
        </div>
      </div>
      <div className="results-box">
        {weightedSum !== null && (
          <>
            <p><strong>Weighted Sum:</strong> (W1*X1) + (W2*X2) + Bias = **{weightedSum}**</p>
            <p className="final-prediction">
              <strong>Final Prediction: </strong> 
              {prediction === 1 ? "YES (1)" : "NO (0)"}
            </p>
            <p className="rule">
                Rule: If Weighted Sum $\ge 0$, then Prediction is 1.
            </p>
          </>
        )}
      </div>
    </div>
  );
};

// ------------------------------------------------------------------
// ScoreBar Component
// ------------------------------------------------------------------
const ScoreBar = ({ label, score, isPrediction }) => {
  // (This component is unchanged)
  const minScore = -2.0;
  const maxScore = 2.0;
  const percent = ((score - minScore) / (maxScore - minScore)) * 100;
  return (
      <div className={`score-bar-container ${isPrediction ? 'prediction-highlight' : ''}`}>
          <div className="score-label">{label}</div>
          <div className="score-bar-track">
              <div 
                  className="score-bar-fill" 
                  style={{ width: `${Math.max(0, Math.min(100, percent))}%` }}
              ></div>
          </div>
          <div className="score-value">{score.toFixed(4)}</div>
      </div>
  );
};

// ------------------------------------------------------------------
// Utility Function
// ------------------------------------------------------------------
const shuffleArray = (array) => {
  // (This function is unchanged)
  let currentIndex = array.length,  randomIndex;
  while (currentIndex != 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
  }
  return array;
}

// ------------------------------------------------------------------
// CNNModel Component
// ------------------------------------------------------------------
const CNNModel = () => {
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [attemptCount, setAttemptCount] = useState(1);
  const [prediction, setPrediction] = useState("Select an image to analyze");
  const [message, setMessage] = useState("Click an image, then click 'Simulate Image Analysis'.");
  const [scores, setScores] = useState([]);
  const [isPredicted, setIsPredicted] = useState(false);
  
  // NEW: This state tracks all the wrong guesses for this round
  const [disprovenClasses, setDisprovenClasses] = useState([]);
  
  // We no longer need isCorrecting
  // const [isCorrecting, setIsCorrecting] = useState(false);

  const [dogImageUrl, setDogImageUrl] = useState(null);
  const [isLoadingDog, setIsLoadingDog] = useState(true);
  const [catImageUrl, setCatImageUrl] = useState(null);
  const [isLoadingCat, setIsLoadingCat] = useState(true);
  const [personImageUrl, setPersonImageUrl] = useState(null);
  const [isLoadingPerson, setIsLoadingPerson] = useState(true);
  const [imageOrder, setImageOrder] = useState([]);
  const baseImageIds = ["cat_image", "dog_image", "person_image"];

  const fetchRandomCat = async () => {
    setIsLoadingCat(true);
    try {
      const response = await fetch(CAT_API_URL, { headers: { 'x-api-key': CAT_API_KEY } });
      const data = await response.json();
      if (data && data[0] && data[0].url) { setCatImageUrl(data[0].url); }
    } catch (error) {
      console.error("Error fetching cat image:", error);
      setCatImageUrl("https://placehold.co/150x150/ff0000/ffffff?text=API+Error");
    }
  };

  const fetchRandomDog = async () => {
    setIsLoadingDog(true);
    try {
      const response = await fetch(DOG_API_URL, { headers: { 'x-api-key': DOG_API_KEY } });
      const data = await response.json();
      if (data && data[0] && data[0].url) { setDogImageUrl(data[0].url); }
    } catch (error) {
      console.error("Error fetching dog image:", error);
      setDogImageUrl("https://placehold.co/150x150/ff0000/ffffff?text=API+Error");
    }
  };

  const fetchRandomPerson = () => {
    setIsLoadingPerson(true);
    const uniqueUrl = `${PERSON_API_URL}?${new Date().getTime()}`;
    setPersonImageUrl(uniqueUrl);
  };

  const fetchAllImages = () => {
    fetchRandomDog();
    fetchRandomCat();
    fetchRandomPerson();
  }

  useEffect(() => {
    fetchAllImages();
    setImageOrder(shuffleArray([...baseImageIds]));
  }, []);

  const resetSelection = () => {
    setSelectedImageId(null);
    setAttemptCount(1);
    setPrediction("Select an image to analyze");
    setMessage("Click an image, then click 'Simulate Image Analysis'.");
    setScores([]);
    setIsPredicted(false);
    
    // NEW: Reset the list of disproven classes
    setDisprovenClasses([]);
    
    fetchAllImages();
    setImageOrder(shuffleArray([...baseImageIds]));
  }

  const handleAnalysis = async () => {
    if (!selectedImageId) {
      setMessage("Please select an image first!");
      return;
    }
    
    // We only set isPredicted(false) on the *first* attempt
    if (attemptCount === 1) {
      setIsPredicted(false);
    }
    
    setMessage("Model is thinking...");
    
    try {
      const response = await fetch("http://localhost:5000/api/cnn/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // NEW: Send the list of classes to ignore
        body: JSON.stringify({ 
          imageId: selectedImageId,
          ignore_classes: disprovenClasses // Pass the list of wrong guesses
        }),
      });
      const data = await response.json();
      setPrediction(data.prediction);
      setScores(data.scores);
      setMessage(`My guess is **${data.prediction}**. Is this correct?`);
      setIsPredicted(true);

    } catch (error) {
      setMessage("Error: Could not get prediction. Is Flask server running?");
      console.error("Prediction Error:", error);
      setIsPredicted(true);
    }
  };

  const handleFeedback = (wasCorrect) => {
    if (!isPredicted) return;

    if (wasCorrect) {
      // --- NEW: This is the "Correct" button logic ---
      setMessage(`✅ Correct! I learned the answer was ${prediction}.`);
      
      // Send the final data to the backend to train
      fetch("http://localhost:5000/api/cnn/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          imageId: selectedImageId,
          correct_class: prediction, // The last guess was the right one
          incorrect_classes: disprovenClasses // The list of all previous wrong guesses
        }),
      });

      // Reset for the next round
      setTimeout(() => {
        resetSelection();
      }, 2500);
      return;

    } else {
      // --- NEW: This is the "Incorrect" button logic ---
      setAttemptCount(prev => prev + 1);
      
      // Add the wrong guess to the list
      const newDisprovenClasses = [...disprovenClasses, prediction];
      setDisprovenClasses(newDisprovenClasses);
      
      // Immediately call handleAnalysis again to make a new guess
      // It will use the newDisprovenClasses list to ignore the wrong guess
      handleAnalysis(); 
    }
  };

  const onImageClick = (imageId) => {
    if (isPredicted) { // Simplified check
        setMessage("Please resolve the current prediction first (Correct/Incorrect).");
        return;
    }
    setSelectedImageId(imageId);
    setAttemptCount(1);
    setScores([]);
    setDisprovenClasses([]); // Reset disproven list on new image click
    setIsPredicted(false);
    setMessage("Image selected. Ready to analyze.");
    setPrediction("Ready to analyze");
  }

  return (
    <div className="cnn-model card">
      <h2>2. The CNN: Interactive Learning (Image Classifier) 👁️‍🗨️</h2>
      <p className="concept-explanation">
        Click one of the images below to select it. Then, press the "Simulate" button to have the AI try and guess what it is.
      </p>
      
      <div className="image-selection-container">
        {imageOrder.map((id) => (
          <div
            key={id}
            className={`selectable-image-container ${selectedImageId === id ? 'active' : 'inactive'}`}
            onClick={() => onImageClick(id)}
          >
            {id === 'cat_image' ? (
              <>
                {isLoadingCat && <div className="image-loader">Loading...</div>}
                <img 
                  src={catImageUrl} alt="Selectable cat_image" className="selectable-image"
                  style={{ visibility: isLoadingCat ? 'hidden' : 'visible' }}
                  onLoad={() => setIsLoadingCat(false)}
                />
              </>
            ) : id === 'dog_image' ? (
              <>
                {isLoadingDog && <div className="image-loader">Loading...</div>}
                <img 
                  src={dogImageUrl} alt="Selectable dog_image" className="selectable-image" 
                  style={{ visibility: isLoadingDog ? 'hidden' : 'visible' }}
                  onLoad={() => setIsLoadingDog(false)}
                />
              </>
            ) : ( // 'person_image'
              <>
                {isLoadingPerson && <div className="image-loader">Loading...</div>}
                <img 
                  src={personImageUrl} alt="Selectable person_image" className="selectable-image" 
                  style={{ visibility: isLoadingPerson ? 'hidden' : 'visible' }}
                  onLoad={() => setIsLoadingPerson(false)}
                />
              </>
            )}
          </div>
        ))}
      </div>
      
      <div className="image-selector">
        <button 
          onClick={handleAnalysis} 
          className="button upload-button" 
          // Simplified disabled logic
          disabled={!selectedImageId || (isPredicted && message.startsWith('✅'))}
        > 
          {/* Change button text based on state */}
          {attemptCount === 1 ? 'Simulate Image Analysis' : 'Guess Again'}
        </button>
      </div>
      
      <h3 style={{marginTop: '20px', color: 'var(--SC)'}}>AI's Prediction:</h3>
      <p className="ai-description-result" style={{backgroundColor: message.startsWith('✅') ? '#ccffcc' : 'var(--card-bg)'}}>
        <strong>{prediction}</strong>
      </p>
      
      <p style={{fontStyle: 'italic', fontSize: '0.9rem', color: 'var(--TC-subtle)'}}>Attempt: {attemptCount}</p>
      
      {/* --- NEW: Simplified Feedback Buttons --- */}
      {/* Show these buttons *only* after a prediction is made and we are *not* yet celebrating */}
      {isPredicted && !message.startsWith('✅') && (
        <div className="feedback-section">
          <p style={{color: 'var(--TC)'}}>Is my guess correct?</p>
          <button className="button button-correct" onClick={() => handleFeedback(true)}>✅ Yes, Correct</button>
          <button className="button button-incorrect" onClick={() => handleFeedback(false)}>❌ No, Incorrect</button>
        </div>
      )}
      
      <p className="status-message">{message}</p>

      <details className="weight-details" open>
        <summary>Current Model Status (Scores)</summary>
        <div style={{padding: '10px'}}>
          <p><strong>Score Breakdown:</strong> (Class with highest score is the prediction)</p>
          {scores.length > 0 ? (
            CLASSES.map((cls, i) => (
              <ScoreBar 
                key={cls}
                label={cls}
                score={scores[i]}
                isPrediction={cls === prediction}
              />
            ))
          ) : (
            <p>No scores yet. Select an image and click 'Simulate'.</p>
          )}
        </div>
      </details>
    </div>
  );
};

// ------------------------------------------------------------------
// Main App Component
// ------------------------------------------------------------------
const App = () => {
    // (This component is unchanged)
    const [message, setMessage] = useState("");
 
    useEffect(() => {
      fetch("http://localhost:5000/api/hello")
        .then((response) => response.json())
        .then((data) => setMessage(data.message))
        .catch((error) => {
            setMessage("Please ensure the Flask backend is running on http://localhost:5000.");
            console.error("Error:", error);
        });
    }, []);
 
    return (
      <div className="App-Container">
        <div className="title-card">
          <h1>Machine Learning Concepts Made Simple 💡</h1>
        </div>
        <div className="message">
          <p>{message}</p>
        </div>
        
        <div className="ai-section">
          <PerceptronModel />
        </div>
 
        <div className="ai-section">
          <CNNModel />
        </div>
 
        <div className="footer">
          <p>Built from scratch to understand AI and Full-Stack Development.</p>
        </div>
      </div>
    );
};

export default App;