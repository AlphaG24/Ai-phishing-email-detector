import os
import sys
import joblib
import json
import pandas as pd
from scipy.sparse import hstack
from flask import Flask, request, jsonify
from flask_cors import CORS 

# Assuming these modules are in the src directory and can be imported
try:
    # Attempt to import from the src directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, 'src'))
    from data_processor import clean_text
    from advanced_features import get_advanced_features
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app) 

# --- Global Model and Vectorizer Loading ---
MODEL = None
VECTORIZER = None
# Define the order of 9 features once, which MUST match the trained model (50009 total features)
FEATURE_COLS = ['num_urls', 'has_suspicious_keywords', 'has_suspicious_title', 
                'has_hidden_images', 'has_suspicious_url_pattern', 
                'has_high_html_ratio', 'has_common_typos', 'has_spoofed_header',
                'is_newly_registered_domain'] # FINAL 9 FEATURES ADDED

def load_assets():
    """Loads the trained model and vectorizer into memory."""
    global MODEL, VECTORIZER
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We use the Naive Bayes model trained with 9 advanced features
    model_path = os.path.join(base_dir, "models", "final", "naive_bayes_model.joblib")
    vectorizer_path = os.path.join(base_dir, "models", "final", "tfidf_vectorizer_advanced.joblib")

    try:
        MODEL = joblib.load(model_path)
        VECTORIZER = joblib.load(vectorizer_path)
        print("Model and Vectorizer loaded successfully.")
    except FileNotFoundError:
        print("FATAL ERROR: Model files not found. Ensure 'python src/train_model.py' has been run.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Could not load assets: {e}")
        sys.exit(1)

def generate_explanation(input_features, prediction):
    """
    Analyzes the extracted features and generates human-readable explanations 
    of why the email was classified as phishing or legitimate.
    """
    reasons = []
    
    is_phishing = prediction == 1
    
    # --- Phishing Signals ---
    
    if input_features.get('has_spoofed_header', 0) == 1:
        reasons.append("🛡️ **Header Spoofing Detected:** The email failed security checks (SPF/DKIM/DMARC), indicating the sender address is forged.")

    if input_features.get('is_newly_registered_domain', 0) == 1:
        reasons.append("📅 **Newly Registered Domain:** The primary URL domain was recently created (Domain Age check failed), a high-risk indicator.")
        
    if input_features.get('has_suspicious_keywords', 0) == 1:
        reasons.append("🚨 **Suspicious Keywords Found:** The email contains high-urgency words (e.g., 'URGENT', 'verify', 'action required'), a common phishing tactic.")

    if input_features.get('num_urls', 0) > 0:
        reasons.append(f"🔗 **URL Presence:** The email contains {input_features['num_urls']} URL(s). Unsolicited links are a primary vector for phishing attacks.")
    
    if input_features.get('has_suspicious_url_pattern', 0) == 1:
        reasons.append("🌐 **Deceptive URL Pattern:** The URL contains suspicious elements (e.g., IP address, non-standard port, excessive subdomains).")
        
    if input_features.get('has_common_typos', 0) == 1:
        reasons.append("📝 **Common Typo Detected:** The email contains misspellings of brand names or common phishing typos (e.g., 'paypaI'), a low-effort attack indicator.")

    if input_features.get('has_suspicious_title', 0) == 1:
        reasons.append("❗ **Generic/Empty Subject:** The email has a generic or blank subject line, which is often used by spammers and phishers.")
        
    if input_features.get('has_high_html_ratio', 0) == 1:
        reasons.append("✉️ **High HTML-to-Text Ratio:** The email's content is heavily formatted with complex HTML, potentially hiding malicious code or making tracking pixels hard to detect.")

    if input_features.get('has_hidden_images', 0) == 1:
        reasons.append("🖼️ **Hidden Image Detected:** The email may contain a 1x1 tracking pixel, often used to confirm the recipient's address is active.")

    # --- Legitimate Signal (If few or no Phishing signals were found) ---
    if not reasons and not is_phishing:
        reasons.append("✅ **Clean Language and Structure:** The email lacks typical phishing markers, suggesting it is a legitimate message.")

    # Return only the top 3 most relevant reasons
    return reasons[:3]


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that accepts email text and returns a prediction and explanation.
    Expected JSON input: {"email_text": "your email content here"}
    """
    if not request.json or 'email_text' not in request.json:
        return jsonify({"error": "Missing 'email_text' in request body."}), 400
    
    email_text = request.json['email_text']

    if MODEL is None or VECTORIZER is None:
        return jsonify({"error": "Model not loaded. Server issue."}), 500

    # Step 1: Clean and extract features
    cleaned_text = clean_text(email_text)
    
    # Step 2: Vectorize text features
    text_vectorized = VECTORIZER.transform([cleaned_text])
    
    # Step 3: Extract advanced numerical features
    advanced_features_dict = get_advanced_features(email_text)
    
    # --- ULTIMATE FIX: Create a guaranteed dictionary for Pandas ---
    # Create a new dictionary based on FEATURE_COLS, ensuring every key exists.
    final_features_for_pandas = {col: advanced_features_dict.get(col, 0) for col in FEATURE_COLS}

    advanced_features_df = pd.DataFrame([final_features_for_pandas])
    
    # Select features in the guaranteed correct order
    advanced_features_ordered = advanced_features_df[FEATURE_COLS]
    
    # Step 4: Combine all features
    X_combined = hstack([text_vectorized, advanced_features_ordered.to_numpy()])

    # Step 5: Predict
    prediction = MODEL.predict(X_combined)[0]
    prediction_proba = MODEL.predict_proba(X_combined)[0]
    
    # Map prediction to label
    label = "PHISHING" if prediction == 1 else "LEGITIMATE (HAM)"
    confidence = float(prediction_proba[prediction])
    
    # Step 6: Generate explanation for the UI
    explanation_reasons = generate_explanation(advanced_features_dict, prediction)


    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "input_features": advanced_features_dict,
        "explanation": explanation_reasons
    })

if __name__ == '__main__':
    load_assets()
    app.run(host='0.0.0.0', port=5000, debug=False)
