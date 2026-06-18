# =========================================================================
# FILE: main_updates.py
# INSTRUCTIONS: 
# Copy and merge this code into your existing `main.py` in the gold-ai-server repo.
# =========================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import csv
import os
from datetime import datetime

# Add this to your existing Pydantic models
class FeedbackData(BaseModel):
    signal: str
    entry_price: float
    result: str # "WIN" or "LOSS" or "BREAKEVEN"
    profit_loss: float = 0.0

# Add this new endpoint to your existing FastAPI app
@app.post("/feedback")
def receive_feedback(data: FeedbackData):
    try:
        feedback_file = 'feedback_log.csv'
        file_exists = os.path.isfile(feedback_file)
        
        with open(feedback_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Create headers if file doesn't exist
                writer.writerow(['timestamp', 'signal', 'entry_price', 'result', 'profit_loss'])
                
            writer.writerow([
                datetime.utcnow().isoformat(),
                data.signal,
                data.entry_price,
                data.result,
                data.profit_loss
            ])
            
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
