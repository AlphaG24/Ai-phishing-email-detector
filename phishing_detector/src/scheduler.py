import schedule
import time
from retrain_model import retrain_with_feedback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scheduled_retraining():
    """Scheduled retraining job"""
    logging.info("🔄 Running scheduled retraining...")
    try:
        success = retrain_with_feedback()
        if success:
            logging.info("✅ Scheduled retraining completed successfully!")
        else:
            logging.info("ℹ️  Scheduled retraining skipped (insufficient data)")
    except Exception as e:
        logging.error(f"❌ Scheduled retraining failed: {e}")

# Schedule retraining every Sunday at 2 AM
schedule.every().sunday.at("02:00").do(scheduled_retraining)

# For testing: run every day
# schedule.every().day.at("02:00").do(scheduled_retraining)

if __name__ == "__main__":
    logging.info("⏰ Retraining scheduler started...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute