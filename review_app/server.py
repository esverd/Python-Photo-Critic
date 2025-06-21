# review_app/server.py
import json
import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory

if len(sys.argv) != 3:
    print("Usage: python server.py <path_to_data_file> <path_to_image_directory>")
    sys.exit(1)

TEMP_DATA_PATH = sys.argv[1]
IMAGE_DIR_PATH = os.path.abspath(sys.argv[2])

app = Flask(__name__, template_folder='.')
app.config['IMAGE_DIR'] = IMAGE_DIR_PATH

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_DIR'], filename)

@app.route('/')
def index():
    if not os.path.exists(TEMP_DATA_PATH):
        return "Error: Data file not found.", 404
    with open(TEMP_DATA_PATH, 'r', encoding='utf-8') as f:
        images_data = json.load(f)
    return render_template('index.html', images=images_data)

@app.route('/save', methods=['POST'])
def save_selection():
    # Use the correct key 'saved_images' to match the front-end
    final_selection = request.json.get('saved_images', [])
    final_selection_path = os.path.join(os.path.dirname(TEMP_DATA_PATH), "final_selection.json")
    with open(final_selection_path, 'w', encoding='utf-8') as f:
        json.dump(final_selection, f, indent=2)
    print(f"✅ {len(final_selection)} image statuses saved to {final_selection_path}")
    return jsonify({"status": "success", "message": "Selection saved."})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return 'Server shutting down...'
    func()
    return 'Server is shutting down...'

if __name__ == '__main__':
    print(f"Starting Flask server for photo review at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
