# retrain_model.py - CORRECTED VERSION
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from scipy.sparse import hstack
import os

# Import your existing functions (make sure these are defined or imported)
from your_main_module import clean_text, get_advanced_features, FEATURE_COLS, VECTORIZER

def retrain_model():
    """Retrain model with new feedback data"""
    try:
        print("🔄 Starting model retraining...")
        
        # 1. Load existing model assets
        MODEL = joblib.load('model.pkl')
        VECTORIZER = joblib.load('vectorizer.pkl')
        
        # 2. Load new feedback data
        if not os.path.exists('feedback_data.csv'):
            print("❌ No feedback data found for retraining")
            return False
            
        feedback_df = pd.read_csv('feedback_data.csv')
        print(f"📊 Loaded {len(feedback_df)} feedback samples")
        
        if len(feedback_df) < 10:  # Minimum samples threshold
            print("⚠️ Not enough feedback samples for meaningful retraining")
            return False
        
        # 3. Prepare features and labels
        X_texts = []
        X_features = []
        y_labels = []
        
        for _, row in feedback_df.iterrows():
            try:
                email_text = row['email_text']
                label = row['correct_label']  # 0 for LEGITIMATE, 1 for PHISHING
                
                # Clean and extract features
                cleaned_text = clean_text(email_text)
                advanced_features = get_advanced_features(email_text)
                
                # Prepare feature vector
                final_features = [advanced_features.get(col, 0) for col in FEATURE_COLS]
                
                X_texts.append(cleaned_text)
                X_features.append(final_features)
                y_labels.append(label)
                
            except Exception as e:
                print(f"⚠️ Skipping problematic feedback sample: {e}")
                continue
        
        # 4. Vectorize and combine features
        text_vectors = VECTORIZER.transform(X_texts)
        feature_arrays = np.array(X_features)
        X_combined = hstack([text_vectors, feature_arrays])
        
        # 5. Retrain model
        MODEL.fit(X_combined, y_labels)
        
        # 6. Save updated model
        joblib.dump(MODEL, 'model.pkl')
        joblib.dump(VECTORIZER, 'vectorizer.pkl')
        
        print("✅ Model retrained successfully with feedback data!")
        return True
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return False

if __name__ == "__main__":
    retrain_model()