import os
import logging
from flask import Flask, request, jsonify
from supabase import create_client, Client

# Import the new initialization and webhook processing functions
from disaster_relief_engine import initialize_models, process_live_webhook

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  |  %(levelname)-8s  |  %(message)s'
)
log = logging.getLogger("flask_app")

app = Flask(__name__)

import threading

models = None
training_started = False
training_lock = threading.Lock()

def init_models_bg():
    global models
    log.info("Starting background thread for ML models initialization...")
    try:
        models = initialize_models()
        log.info("Models successfully held in memory.")
    except Exception as e:
        log.error(f"Failed to initialize models on startup: {e}")


# Initialize Supabase Client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = None
if not supabase_url or not supabase_key:
    log.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables are missing! "
                "Database operations will fail.")
else:
    supabase = create_client(supabase_url, supabase_key)
    log.info("Supabase client initialized.")


# Webhook Endpoint
@app.route("/webhook/match", methods=["POST"])
def match_webhook():
    """
    Supabase webhook receiver for the `needs` table.
    Triggers when a new Need is inserted.
    """
    try:
        payload = request.json
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Step 1: Parse the incoming JSON.
        # Supabase webhooks send the new row inside 'record' for INSERT events.
        new_need = payload.get("record")
        if not new_need:
            return jsonify({"error": "Missing 'record' in payload"}), 400

        need_id = new_need.get("id")
        zone = new_need.get("zone")

        log.info(f"Received webhook for Need ID: {need_id} in Zone: {zone}")

        if not supabase:
            return jsonify({"error": "Supabase client is not configured"}), 500

        if not models:
            return jsonify({"error": "ML models are not initialized"}), 500

        # Step 2: Use Supabase client to fetch all available volunteers
        # (active=True AND zone matches the new need's zone)
        volunteers_response = supabase.table("volunteers") \
            .select("*") \
            .eq("active", True) \
            .eq("zone", zone) \
            .execute()
        
        active_volunteers = volunteers_response.data
        if not active_volunteers:
            log.info(f"No active volunteers found in zone {zone} for need {need_id}.")
            return jsonify({"success": True, "message": "No available volunteers in zone", "assignments": []}), 200

        # Step 3: Pass new need and fetched volunteers to the live webhook processor
        assignments = process_live_webhook(new_need, active_volunteers, models)

        if not assignments:
            log.info(f"No optimal assignments could be generated for need {need_id}.")
            return jsonify({"success": True, "message": "No assignments could be made", "assignments": []}), 200

        log.info(f"Generated {len(assignments)} assignments for need {need_id}.")

        # Step 4: Use Supabase client to INSERT the resulting assignments
        supabase.table("assignments").insert(assignments).execute()

        # Step 5: Use Supabase client to UPDATE the original need's status to "assigned"
        supabase.table("needs").update({"status": "assigned"}).eq("id", need_id).execute()

        # Step 6: Return a 200 JSON success response
        return jsonify({
            "success": True,
            "message": "Matching completed successfully",
            "assignments": assignments
        }), 200

    except Exception as e:
        log.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint for Render."""
    global training_started
    
    # Trigger background training gracefully on first hit
    if models is None:
        with training_lock:
            if not training_started:
                training_started = True
                log.info("Lazy initialization triggered by /health endpoint...")
                threading.Thread(target=init_models_bg, daemon=True).start()

    return jsonify({
        "status": "up",
        "models_loaded": models is not None,
        "supabase_configured": supabase is not None
    }), 200


if __name__ == "__main__":
    # Use port 5000 or the port provided by the environment (Render uses PORT)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
