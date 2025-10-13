from typing import List
from pydantic import BaseModel

class BulkEmailInput(BaseModel):
    emails: List[str]

class BulkPredictionResult(BaseModel):
    email_preview: str
    prediction: str
    confidence: float
    features: dict

@app.post("/bulk_predict")
async def bulk_predict_emails(data: BulkEmailInput):
    """Bulk analysis endpoint for multiple emails"""
    if not GLOBAL_LOAD_SUCCESS:
        raise HTTPException(status_code=500, detail="AI Model assets failed to load.")
    
    results = []
    
    for i, email_text in enumerate(data.emails):
        try:
            # Clean and extract features
            cleaned_text = clean_text(email_text)
            advanced_features_dict = get_advanced_features(email_text)
            
            # Prepare features
            final_features = {col: advanced_features_dict.get(col, 0) for col in FEATURE_COLS}
            
            # Vectorize and predict
            text_vectorized = VECTORIZER.transform([cleaned_text])
            advanced_features_df = pd.DataFrame([final_features])
            advanced_features_array = advanced_features_df[FEATURE_COLS].to_numpy()
            X_combined = hstack([text_vectorized, advanced_features_array])
            
            prediction = MODEL.predict(X_combined)[0]
            prediction_proba = MODEL.predict_proba(X_combined)[0]
            confidence = float(prediction_proba[prediction])
            
            result = BulkPredictionResult(
                email_preview=email_text[:100] + "..." if len(email_text) > 100 else email_text,
                prediction="PHISHING" if prediction == 1 else "LEGITIMATE",
                confidence=round(confidence, 4),
                features=advanced_features_dict
            )
            results.append(result)
            
        except Exception as e:
            # Skip problematic emails but continue processing others
            print(f"⚠️ Error processing email {i}: {e}")
            continue
    
    return {
        "total_processed": len(results),
        "phishing_count": sum(1 for r in results if r.prediction == "PHISHING"),
        "legitimate_count": sum(1 for r in results if r.prediction == "LEGITIMATE"),
        "results": results
    }