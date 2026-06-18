# =========================================================================
# FILE: main_updates.py
# INSTRUCTIONS: 
# Copy and merge this code into your existing `app.py` or `main.py` (Flask app) 
# in the gold-ai-server repo.
# =========================================================================

from flask import request, jsonify
import csv
import os
from datetime import datetime

# Assuming your Flask app is named `app`
# app = Flask(__name__)

# Add this new route to your existing Flask app
@app.route("/feedback", methods=["POST"])
def receive_feedback():
    try:
        data = request.get_json()
        
        if not data or 'signal' not in data or 'result' not in data:
            return jsonify({"status": "error", "message": "Invalid payload"}), 400
            
        signal = data.get('signal')
        entry_price = data.get('entry_price', 0.0)
        result = data.get('result')
        profit_loss = data.get('profit_loss', 0.0)

        feedback_file = 'feedback_log.csv'
        file_exists = os.path.isfile(feedback_file)
        
        with open(feedback_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # Create headers if file doesn't exist
                writer.writerow(['timestamp', 'signal', 'entry_price', 'result', 'profit_loss'])
                
            writer.writerow([
                datetime.utcnow().isoformat(),
                signal,
                entry_price,
                result,
                profit_loss
            ])
            
        return jsonify({"status": "success", "message": "Feedback recorded successfully."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
