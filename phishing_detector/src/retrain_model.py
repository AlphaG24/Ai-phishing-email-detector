import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack
import os
import sys

sys.path.append(os.path.dirname(__file__))
from advanced_features import get_advanced_features
from train_model import clean_text, feature_cols, vectorizer

FEEDBACK_FILE = 'feedback_data.csv'
MODEL_FILE = "models/final/naive_bayes_model.joblib"
VECTORIZER_FILE = "models/final/tfidf_vectorizer_advanced.joblib"
HISTORY_FILE = "models/final/retrained_ids.csv"  # Track already used feedback

def retrain_model():
    """Retrain model with new feedback data"""
    try:
        print("🔄 Starting model retraining...")

        # Load existing model and vectorizer
        MODEL = joblib.load(MODEL_FILE)
        VECTORIZER = joblib.load(VECTORIZER_FILE)

        # Load feedback
        if not os.path.exists(FEEDBACK_FILE):
            print("❌ No feedback data found for retraining")
            return False

        feedback_df = pd.read_csv(FEEDBACK_FILE)
        print(f"📊 Loaded {len(feedback_df)} feedback samples")

        # Load previously retrained IDs (optional, if you have unique IDs)
        if os.path.exists(HISTORY_FILE):
            used_ids = pd.read_csv(HISTORY_FILE)['text'].tolist()
        else:
            used_ids = []

        # Filter out already used feedback
        feedback_df = feedback_df[~feedback_df['text'].isin(used_ids)]

        if len(feedback_df) < 1:  # Need at least 1 new sample
            print("⚠️ No new feedback to retrain on")
            return False

        print(f"📌 {len(feedback_df)} new feedback samples to retrain")

        X_texts, X_features, y_labels = [], [], []

        for _, row in feedback_df.iterrows():
            email_text = row['text']
            label = row['label']  # 0 for LEGITIMATE, 1 for PHISHING

            cleaned_text = clean_text(email_text)
            adv_features = get_advanced_features(email_text)
            final_features = [adv_features.get(col, 0) for col in feature_cols]

            X_texts.append(cleaned_text)
            X_features.append(final_features)
            y_labels.append(label)

        # Vectorize and combine features
        text_vectors = VECTORIZER.transform(X_texts)
        feature_arrays = np.array(X_features)
        X_combined = hstack([text_vectors, feature_arrays])

        # Retrain model
        MODEL.fit(X_combined, y_labels)

        # Save updated model
        joblib.dump(MODEL, MODEL_FILE)
        joblib.dump(VECTORIZER, VECTORIZER_FILE)

        # Update history file
        pd.DataFrame({'text': feedback_df['text']}).to_csv(HISTORY_FILE, index=False)

        print("✅ Model retrained successfully with new feedback!")
        return True

    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        return False

if __name__ == "__main__":
    retrain_model()
