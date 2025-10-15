import os
import sys
import joblib
import pandas as pd
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import csv
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import io
import re
import logging
import gdown


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Google Drive model downloader ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "final")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILES = {
    "naive_bayes_model.joblib": "1L34TIsh6dsnYK8CVV9hn5QdBrnnPcU3H",
    "semantic_model.joblib": "1KMQYQgGI5cWori75eYlanKR4P8gSEbyl",
    "tfidf_vectorizer_advanced.joblib": "1oP_Jy8Da-2zI-R4-VEq5MGhMarrZO92x",
}

for filename, file_id in MODEL_FILES.items():
    file_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(file_path):
        logger.info(f"⬇️ Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(base_dir, 'src'))
    from data_processor import clean_text
    from advanced_features import get_advanced_features
except ImportError as e:
    print(f"FATAL ERROR: Could not import core modules: {e}")
    sys.exit(1)

# --- Configuration ---
FEATURE_COLS = [
    'num_urls', 'has_suspicious_keywords', 'has_suspicious_title',
    'has_hidden_images', 'has_suspicious_url_pattern',
    'has_high_html_ratio', 'has_common_typos', 'has_spoofed_header',
    'is_newly_registered_domain', 'has_typosquatting_link',
    'has_suspicious_tld', 'has_url_obfuscation', 'has_urgency_language',
    'has_poor_grammar'
]
FEEDBACK_FILE = 'feedback_data.csv'

# --- Pydantic Models ---
class EmailInput(BaseModel):
    email_text: str

class FeedbackInput(BaseModel):
    email_text: str
    label: int

class BulkEmailInput(BaseModel):
    emails: List[str]

# --- Global Model and Vectorizer ---
MODEL = None
VECTORIZER = None
GLOBAL_LOAD_SUCCESS = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VECTORIZER, GLOBAL_LOAD_SUCCESS
    print("⬇️ Checking for required NLTK data...")
    try:
        import nltk
        # Create a path for NLTK data within your project
        nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
        os.makedirs(nltk_data_path, exist_ok=True)
        # Add the path to NLTK's data path list
        nltk.data.path.append(nltk_data_path)
        
        # Download the 'punkt' tokenizer data if not already present
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        print("✅ NLTK 'punkt' data is ready.")
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
    print("🔄 Starting up... Loading model and vectorizer")
    try:
        MODEL = joblib.load(os.path.join(base_dir, "models", "final", "naive_bayes_model.joblib"))
        VECTORIZER = joblib.load(os.path.join(base_dir, "models", "final", "tfidf_vectorizer_advanced.joblib"))
        GLOBAL_LOAD_SUCCESS = True
        print("✅ Model and Vectorizer loaded successfully (50,014 features).")
    except FileNotFoundError:
        print("❌ Model files not found. Run 'python src/train_model.py'.")
    except Exception as e:
        print(f"❌ Could not load assets: {e}")
    yield
    print("🔴 Shutting down...")

# --- FastAPI App ---
app = FastAPI(title="AI Phishing Detection Service", version="1.0.0", lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import Response


# --- Static Files ---
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

# --- Serve SPA frontend properly ---
# Mount the templates folder as a StaticFiles app with html=True
app.mount("/", StaticFiles(directory=os.path.join(base_dir, "templates"), html=True), name="frontend")

# Support Render health check HEAD request
@app.head("/")
async def head_frontend():
    return Response(status_code=200)

# Optional: favicon route
@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(base_dir, "static", "favicon.ico"))

# --- Utility: Explanation Generator ---
def generate_explanation(input_features: Dict[str, Any], prediction: int) -> List[str]:
    reasons = []
    is_phishing = prediction == 1
    if input_features.get('has_spoofed_header', 0): reasons.append("🛡️ Header Spoofing Detected")
    if input_features.get('is_newly_registered_domain', 0): reasons.append("📅 Newly Registered Domain")
    if input_features.get('has_typosquatting_link', 0): reasons.append("🔠 Typosquatting/Homoglyph Detected")
    if input_features.get('has_suspicious_keywords', 0): reasons.append("🚨 Suspicious Keywords Found")
    if input_features.get('has_common_typos', 0): reasons.append("📝 Common Typo Detected")
    if input_features.get('has_url_obfuscation', 0): reasons.append("🔗 URL Obfuscation Detected")
    if input_features.get('has_suspicious_tld', 0): reasons.append("🌐 Suspicious TLD Detected")
    if input_features.get('has_urgency_language', 0): reasons.append("⏰ Urgency Language Detected")
    if input_features.get('has_poor_grammar', 0): reasons.append("📝 Poor Grammar Detected")
    if input_features.get('num_urls', 0) > 0: reasons.append(f"🔗 Contains {input_features['num_urls']} URL(s)")
    if input_features.get('has_suspicious_url_pattern', 0): reasons.append("🌐 Deceptive URL Pattern")
    if not reasons and not is_phishing: reasons.append("✅ Clean Language and Structure")
    return reasons[:3]

# --- Predict (single email) ---
@app.post("/predict")
async def predict_email(data: EmailInput):
    email_text = data.email_text
    
    # Check if this appears to contain multiple emails
    subject_count = email_text.count("Subject:")
    if subject_count > 1:
        return {
        "prediction": "ERROR",
        "message": f"It looks like you pasted multiple emails. Please provide **one email at a time** in this section or use the **Bulk Analysis** feature.",
        "found_emails": subject_count
            }
    if not GLOBAL_LOAD_SUCCESS:
        raise HTTPException(status_code=500, detail="AI Model assets failed to load.")
    
    try:
        cleaned_text = clean_text(data.email_text)
        advanced_features_dict = get_advanced_features(data.email_text)
        final_features = {col: advanced_features_dict.get(col, 0) for col in FEATURE_COLS}
        text_vectorized = VECTORIZER.transform([cleaned_text])
        advanced_features_array = pd.DataFrame([final_features])[FEATURE_COLS].to_numpy()
        X_combined = hstack([text_vectorized, advanced_features_array])
        prediction = MODEL.predict(X_combined)[0]
        prediction_proba = MODEL.predict_proba(X_combined)[0]
        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = float(prediction_proba[prediction])
        return {
            "prediction": label,
            "confidence": round(confidence, 4),
            "input_features": advanced_features_dict,
            "explanation": generate_explanation(advanced_features_dict, prediction)
        }
    except Exception as e:
        logger.error(f"Error in single email prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing email: {str(e)}")

# --- Feedback Endpoint ---
@app.post("/feedback")
async def receive_feedback(data: FeedbackInput):
    new_data = {'text': data.email_text, 'label': data.label, 'timestamp': pd.Timestamp.now().isoformat()}
    try:
        file_exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_data.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(new_data)
        return {"status": "success", "message": "Feedback recorded for retraining."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {e}")

# --- Bulk Upload (TXT/CSV file) ---
@app.post("/bulk_upload")
async def bulk_upload_file(file: UploadFile = File(...)):
    if not GLOBAL_LOAD_SUCCESS:
        raise HTTPException(status_code=500, detail="AI Model assets not loaded.")
    if not file.filename.endswith(('.txt', '.csv')):
        raise HTTPException(status_code=400, detail="Only TXT and CSV files are supported")
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        emails = []

        if file.filename.endswith('.txt'):
            # First try: split on lines that start with "Subject:" (keeps each email block together)
            emails = [e.strip() for e in re.split(r'(?=^Subject:)', content_str, flags=re.M) if e.strip()]

            # Fallback: split by two or more newlines (blank-line separation)
            if not emails or len(emails) == 1:
                emails = [e.strip() for e in re.split(r'\n\s*\n+', content_str) if e.strip()]

        elif file.filename.endswith('.csv'):
            csv_file = io.StringIO(content_str)
            emails = [row[0].strip() for row in csv.reader(csv_file) if row and row[0].strip()]

        # If header detected (first row is a header), drop it
        if emails and any(k in emails[0].lower() for k in ['email', 'text', 'content']):
            emails = emails[1:]

        if not emails:
            raise HTTPException(status_code=400, detail="No valid emails found in file")

        results = []
        phishing_count = legitimate_count = error_count = 0

        for i, email_text in enumerate(emails):
            if not email_text or not str(email_text).strip():
                continue
            try:
                cleaned_text = clean_text(email_text)
                advanced_features_dict = get_advanced_features(email_text)
                final_features = {col: advanced_features_dict.get(col, 0) for col in FEATURE_COLS}
                text_vectorized = VECTORIZER.transform([cleaned_text])
                advanced_features_array = pd.DataFrame([final_features])[FEATURE_COLS].to_numpy()
                X_combined = hstack([text_vectorized, advanced_features_array])
                prediction = MODEL.predict(X_combined)[0]
                prediction_proba = MODEL.predict_proba(X_combined)[0]
                label = "PHISHING" if prediction == 1 else "LEGITIMATE"
                confidence = float(prediction_proba[prediction])
                
                if prediction == 1:
                    phishing_count += 1
                else:
                    legitimate_count += 1

                results.append({
                    "email_preview": email_text[:200] + ("..." if len(email_text) > 200 else ""),
                    "prediction": label,
                    "confidence": confidence,
                    "features": advanced_features_dict,
                    "status": "success"
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing email {i}: {e}")
                error_count += 1
                results.append({
                    "email_preview": email_text[:200] + ("..." if len(email_text) > 200 else ""),
                    "prediction": "ERROR",
                    "confidence": 0.0,
                    "features": {},
                    "status": "error",
                    "error_message": str(e)
                })
                continue

        return {
            "total_processed": len(results),
            "successful_processing": len(results) - error_count,
            "failed_processing": error_count,
            "phishing_count": phishing_count,
            "legitimate_count": legitimate_count,
            "results": results,
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# --- Bulk Predict (pasted emails) ---
@app.post("/bulk_predict")
async def bulk_predict(data: BulkEmailInput):
    if not GLOBAL_LOAD_SUCCESS:
        raise HTTPException(status_code=500, detail="Model assets not loaded.")
    
    results = []
    phishing_count = legitimate_count = error_count = 0

    for i, email_text in enumerate(data.emails):
        if not email_text or not str(email_text).strip():
            continue
        try:
            cleaned_text = clean_text(email_text)
            advanced_features_dict = get_advanced_features(email_text)
            final_features = {col: advanced_features_dict.get(col, 0) for col in FEATURE_COLS}
            text_vectorized = VECTORIZER.transform([cleaned_text])
            advanced_features_array = pd.DataFrame([final_features])[FEATURE_COLS].to_numpy()
            X_combined = hstack([text_vectorized, advanced_features_array])
            prediction = MODEL.predict(X_combined)[0]
            prediction_proba = MODEL.predict_proba(X_combined)[0]
            label = "PHISHING" if prediction == 1 else "LEGITIMATE"
            confidence = float(prediction_proba[prediction])
            
            if prediction == 1:
                phishing_count += 1
            else:
                legitimate_count += 1

            results.append({
                "email_preview": email_text[:200] + ("..." if len(email_text) > 200 else ""),
                "prediction": label,
                "confidence": confidence,
                "features": advanced_features_dict,
                "status": "success"
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing email {i}: {e}")
            error_count += 1
            results.append({
                "email_preview": email_text[:200] + ("..." if len(email_text) > 200 else ""),
                "prediction": "ERROR",
                "confidence": 0.0,
                "features": {},
                "status": "error",
                "error_message": str(e)
            })
            continue

    return {
        "total_processed": len(results),
        "successful_processing": len(results) - error_count,
        "failed_processing": error_count,
        "phishing_count": phishing_count,
        "legitimate_count": legitimate_count,
        "results": results
    }

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "healthy" if GLOBAL_LOAD_SUCCESS else "unhealthy",
            "model_loaded": GLOBAL_LOAD_SUCCESS,
            "features": len(FEATURE_COLS)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))