import os
import sys
import joblib
import pandas as pd
from scipy.sparse import hstack
from data_processor import clean_text
from advanced_features import get_advanced_features

def main():
    """
    Main function to execute the prediction process.
    """
    if len(sys.argv) < 2:
        print("Error: Please provide an email text to classify.")
        print("Usage: python predict.py \"<email text>\"")
        sys.exit(1)
        
    email_text = sys.argv[1]

    # Dynamically find the project root path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Step 1: Loading the trained model and vectorizer...")
    model_path = os.path.join(base_dir, "..", "models", "final", "logistic_regression_model_advanced.joblib")
    vectorizer_path = os.path.join(base_dir, "..", "models", "final", "tfidf_vectorizer_advanced.joblib")

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        print("Error: Model or vectorizer file not found.")
        print("Please ensure you have run 'python train_model.py' to create the model.")
        sys.exit(1)

    print("\nStep 2: Cleaning the input email text and extracting features...")
    cleaned_text = clean_text(email_text)
    
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Extract advanced features from the email
    advanced_features_dict = get_advanced_features(email_text)
    advanced_features_df = pd.DataFrame([advanced_features_dict])
    
    # Combine the vectorized text features with the advanced numerical features
    X_combined = hstack([text_vectorized, advanced_features_df.to_numpy()])

    print("\nStep 3: Making a prediction...")
    prediction = model.predict(X_combined)
    prediction_proba = model.predict_proba(X_combined)[0]

    print("\nPrediction Result:")
    if prediction[0] == 1:
        print("This is a PHISHING email.")
    else:
        print("This is a LEGITIMATE (HAM) email.")
    
    print(f"Confidence (Probability): {prediction_proba[prediction[0]]:.2f}")

if __name__ == "__main__":
    main()
