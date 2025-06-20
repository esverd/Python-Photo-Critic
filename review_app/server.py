import json
import os
from flask import Flask, jsonify, request, send_from_directory
import logging

# Disable Flask's default logging to keep the console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, static_folder='static')

# In-memory data store
IMAGE_DATA = []

def load_data():
    global IMAGE_DATA
    try:
        # The script now looks for the data file in the parent directory
        with open('../temp_data.json', 'r', encoding='utf-8') as f:
            IMAGE_DATA = json.load(f)
    except FileNotFoundError:
        print("Error: temp_data.json not found in the parent directory.")
        IMAGE_DATA = []

@app.route('/')
def index():
    # Serve the HTML file as requested by the user
    return send_from_directory('static', 'photo_browser.html')

@app.route('/api/images', methods=['GET'])
def get_images():
    return jsonify(IMAGE_DATA)

@app.route('/api/update_status/<filename>', methods=['POST'])
def update_status(filename):
    data = request.get_json()
    new_status = data.get('is_approved')

    for img in IMAGE_DATA:
        if img['filename'] == filename:
            img['is_approved'] = new_status
            print(f"Updated {filename}: is_approved = {new_status}")
            return jsonify({"success": True, "message": f"Status updated for {filename}"})

    return jsonify({"success": False, "message": "Image not found"}), 404

def shutdown_server():
    # This function will be called to shut down the server
    os._exit(0)

@app.route('/api/finalize', methods=['POST'])
def finalize_selections():
    print("Finalizing selections...")
    approved_images = [img for img in IMAGE_DATA if img.get('is_approved', False)]
    
    # Save the final selection in the parent directory for the notebook to access
    with open('../final_selection.json', 'w', encoding='utf-8') as f:
        json.dump(approved_images, f, indent=4)
        
    print(f"Saved {len(approved_images)} approved images to final_selection.json.")
    
    shutdown_server()
    
    return jsonify({"success": True, "message": "Selections finalized and server is shutting down."})

if __name__ == '__main__':
    load_data()
    print("Flask server is running. Open http://127.0.0.1:5000 in your browser.")
    app.run(port=5000, debug=False)