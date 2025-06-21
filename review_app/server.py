# review_app/server.py
import json
import os
import sys
from flask import Flask, render_template, request, jsonify

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 3:
    print("Usage: python server.py <path_to_data_file> <path_to_image_directory>")
    sys.exit(1)

# Get the file paths from command-line arguments
TEMP_DATA_PATH = sys.argv[1]
IMAGE_DIR_PATH = sys.argv[2]

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')
app.config['IMAGE_DIR'] = IMAGE_DIR_PATH

# Suppress Flask's default logging to keep the notebook output clean
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# --- Main Route to Display the Review Page ---
@app.route('/')
def index():
    """
    Renders the main review page.
    Reads the initial data from the JSON file created by the notebook.
    """
    if not os.path.exists(TEMP_DATA_PATH):
        return "Error: Data file not found.", 404

    with open(TEMP_DATA_PATH, 'r', encoding='utf-8') as f:
        images_data = json.load(f)

    # Pass the image data and the image directory path to the template
    return render_template('index.html', images=images_data, image_dir=app.config['IMAGE_DIR'])


# --- Route to Save the Final Selections ---
@app.route('/save', methods=['POST'])
def save_selection():
    """
    Receives the final, human-curated selections from the web UI,
    saves them to 'final_selection.json', and returns a success message.
    """
    final_selection = request.json.get('approved_images', [])
    
    # Define the output path for the final selections
    final_selection_path = os.path.join(os.path.dirname(TEMP_DATA_PATH), "final_selection.json")

    with open(final_selection_path, 'w', encoding='utf-8') as f:
        json.dump(final_selection, f, indent=2)
        
    print(f"✅ {len(final_selection)} images approved and saved to {final_selection_path}")
    return jsonify({"status": "success", "message": "Selection saved."})


# --- Route to Shutdown the Server ---
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Shuts down the Flask server. Called from the web UI after saving.
    """
    print("Server shutting down...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'


if __name__ == '__main__':
    print("Starting Flask server for photo review...")
    # Make the server accessible only from your local machine
    app.run(host='127.0.0.1', port=5000, debug=False)