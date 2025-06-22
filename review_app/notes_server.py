# review_app/notes_server.py
import json
import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging

# --- Server Setup ---
# Expects three arguments: temp data file, final data file, and image directory
if len(sys.argv) != 4:
    print("Usage: python notes_server.py <path_to_temp_data> <path_to_final_data> <path_to_image_directory>")
    sys.exit(1)

TEMP_DATA_PATH = sys.argv[1]
FINAL_DATA_PATH = sys.argv[2]
IMAGE_DIR_PATH = os.path.abspath(sys.argv[3])

app = Flask(__name__, template_folder='.')
app.config['IMAGE_DIR'] = IMAGE_DIR_PATH

# Suppress standard Flask logging to keep the notebook output clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Routes ---

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serves the actual image files to the frontend."""
    return send_from_directory(app.config['IMAGE_DIR'], filename)

@app.route('/')
def index():
    """Renders the main notes application page."""
    if not os.path.exists(TEMP_DATA_PATH):
        return "Error: Data file not found.", 404
    with open(TEMP_DATA_PATH, 'r', encoding='utf-8') as f:
        images_data = json.load(f)
    # Note: The template file is now 'notes_app.html'
    return render_template('notes_app.html', images=images_data)

@app.route('/save', methods=['POST'])
def save_notes():
    """Receives the updated data from the frontend and saves it."""
    # The key 'images_with_notes' must match what the frontend sends
    final_data = request.json.get('images_with_notes', [])
    
    # Write to the final output path specified by the notebook
    with open(FINAL_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2)
        
    print(f"✅ {len(final_data)} image notes saved to {FINAL_DATA_PATH}")
    return jsonify({"status": "success", "message": "Notes saved."})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shuts down the Flask server."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # This handles cases where the server is run in a different environment
        return 'Server shutting down...'
    func()
    return 'Server is shutting down...'

if __name__ == '__main__':
    print(f"Starting Flask server for note taking at http://127.0.0.1:5001")
    # Using a different port (5001) to avoid conflicts with the review app
    app.run(host='127.0.0.1', port=5001, debug=False)