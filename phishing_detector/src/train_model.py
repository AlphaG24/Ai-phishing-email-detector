import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.sparse import hstack
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processor import clean_text
from advanced_features import get_advanced_features

feature_cols = [
    'num_urls', 'has_suspicious_keywords', 'has_suspicious_title', 
    'has_hidden_images', 'has_suspicious_url_pattern', 
    'has_high_html_ratio', 'has_coammon_typos', 'has_spoofed_header',
    'is_newly_registered_domain', 'has_typosquatting_link',
    'has_suspicious_tld', 'has_url_obfuscation', 'has_urgency_language', 'has_poor_grammar'
]

vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')

def train_mnb_model():
    """
    Train Multinomial Naive Bayes model with TF-IDF + 10 advanced features
    """
    print("🚀 Starting MNB Model Training...")
    
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, "..", "data", "processed", "cleaned_dataset.csv")
    
    if not os.path.exists(processed_data_path):
        print("❌ Error: Processed data not found. Run data_processor.py first.")
        return
    
    print("📊 Step 1: Loading processed data...")
    df = pd.read_csv(processed_data_path)
    
    # Clean text data
    print("🧹 Step 2: Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Extract advanced features
    print("🔍 Step 3: Extracting 14 advanced features...")
    advanced_features_list = df['text'].apply(get_advanced_features).tolist()
    advanced_features_df = pd.DataFrame(advanced_features_list)
    
    # Define your 10 feature columns
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in advanced_features_df.columns:
            advanced_features_df[col] = 0
            print(f"⚠️  Warning: {col} not found, filling with zeros")
    
    print("🔡 Step 4: Creating TF-IDF features...")
    # Create TF-IDF features from text (50,000 features)
    
    X_text = vectorizer.fit_transform(df['cleaned_text'])
    
    # Get advanced features
    X_advanced = advanced_features_df[feature_cols].values
    
    # Combine TF-IDF (50,000) + Advanced Features (10) = 50,010 total features
    X_combined = hstack([X_text, X_advanced])
    y = df['label']
    
    print(f"📈 Final feature shape: {X_combined.shape}")
    print(f"   - TF-IDF features: {X_text.shape[1]}")
    print(f"   - Advanced features: {X_advanced.shape[1]}")
    print(f"   - Total features: {X_combined.shape[1]}")
    
    # Split data
    print("✂️  Step 5: Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Multinomial Naive Bayes
    print("🎯 Step 6: Training Multinomial Naive Bayes model...")
    mnb_model = MultinomialNB(alpha=0.1)
    mnb_model.fit(X_train, y_train)
    
    # Evaluate model
    print("📊 Step 7: Evaluating model performance...")
    y_pred = mnb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("🤖 MNB MODEL PERFORMANCE")
    print("="*50)
    print(f"✅ Accuracy:  {accuracy:.4f}")
    print(f"🎯 Precision: {precision:.4f}")
    print(f"📈 Recall:    {recall:.4f}")
    print(f"⚡ F1-Score:  {f1:.4f}")
    print("="*50)
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Save model and vectorizer
    print("💾 Step 8: Saving model and vectorizer...")
    models_dir = os.path.join(base_dir, "..", "models", "final")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save MNB model (not logistic regression!)
    model_path = os.path.join(models_dir, "naive_bayes_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer_advanced.joblib")
    
    joblib.dump(mnb_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"✅ MNB Model saved: {model_path}")
    print(f"✅ Vectorizer saved: {vectorizer_path}")
    print(f"✅ Total features used: {X_combined.shape[1]}")
    
    # 🔍 Advanced Feature Analysis:
    print("\n🔍 Advanced Feature Analysis:")
    feature_analysis = pd.DataFrame({
    'feature': feature_cols,
    'mean_value': X_advanced.mean(axis=0)
    })
    feature_analysis = feature_analysis.sort_values('mean_value', ascending=False)

    # Simple aligned columns
    print("Feature".ljust(45) + "Mean Value")
    print("-" * 60)
    for _, row in feature_analysis.iterrows():
        print(f"{row['feature']:<45} {row['mean_value']:>10.6f}")
if __name__ == "__main__":
    train_mnb_model()