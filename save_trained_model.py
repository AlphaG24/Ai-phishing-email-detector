import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.sparse import hstack

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'phishing_detector', 'src'))
from data_processor import clean_text
from advanced_features import get_advanced_features

def save_trained_model():
    """
    Train and save Multinomial Naive Bayes model with TF-IDF + advanced features
    """
    print("🚀 Starting model training...")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "phishing_detector", "data", "processed", "cleaned_dataset.csv")
    models_dir = os.path.join(base_dir, "phishing_detector", "models", "final")
    os.makedirs(models_dir, exist_ok=True)

    # Check data
    if not os.path.exists(data_path):
        print(f"❌ Error: Processed dataset not found at {data_path}")
        return

    # Load dataset
    print("📊 Loading processed dataset...")
    df = pd.read_csv(data_path)

    # Clean text
    print("🧹 Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Extract advanced features
    print("🔍 Extracting advanced features...")
    advanced_features_list = df['text'].apply(get_advanced_features).tolist()
    advanced_features_df = pd.DataFrame(advanced_features_list)

    # Define feature columns
    feature_cols = [
        'num_urls', 'has_suspicious_keywords', 'has_suspicious_title', 
        'has_hidden_images', 'has_suspicious_url_pattern', 
        'has_high_html_ratio', 'has_common_typos', 'has_spoofed_header',
        'is_newly_registered_domain', 'has_typosquatting_link',
        'has_suspicious_tld', 'has_url_obfuscation', 'has_urgency_language', 'has_poor_grammar'
    ]

    # Ensure all features exist
    for col in feature_cols:
        if col not in advanced_features_df.columns:
            advanced_features_df[col] = 0
            print(f"⚠️  Warning: {col} missing, filling with zeros")

    # TF-IDF
    print("🔡 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
    X_text = vectorizer.fit_transform(df['cleaned_text'])

    # Combine with advanced features
    X_advanced = advanced_features_df[feature_cols].to_numpy()
    X_combined = hstack([X_text, X_advanced])
    y = df['label']

    print(f"📈 Feature shape: {X_combined.shape}")

    # Train/test split
    print("✂️  Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("🎯 Training Multinomial Naive Bayes model...")
    mnb_model = MultinomialNB(alpha=0.1)
    mnb_model.fit(X_train, y_train)

    # Evaluate
    print("📊 Evaluating model...")
    y_pred = mnb_model.predict(X_test)
    print("\n" + "="*50)
    print("🤖 MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    # Save model & vectorizer
    model_path = os.path.join(models_dir, "naive_bayes_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer_advanced.joblib")
    joblib.dump(mnb_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Vectorizer saved: {vectorizer_path}")

if __name__ == "__main__":
    save_trained_model()
    print("✅ Training complete and model saved successfully!")
