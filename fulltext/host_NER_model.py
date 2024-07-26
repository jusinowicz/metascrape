#==============================================================================
# This file is to host the NER as an app using flask so that it can be run with 
# Label Studio.
#==============================================================================
#libraries
from flask import Flask, request, jsonify
import spacy
import sys
import os

#the custom modules
sys.path.append(os.path.abspath('./../'))  
from common.config import load_config, get_config_param, ConfigError
#==============================================================================
# Load the custom model
config_file_path = './config_abstracts.csv'
try:
    config = load_config(config_file_path)
    model_load_dir = get_config_param(config, 'model_load_dir', required=True)
except ConfigError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

nlp = spacy.load(model_load_dir)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    if text:
        doc = nlp(text)
        predictions = []
        for ent in doc.ents:
            predictions.append({
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })
        return jsonify(predictions)
    return jsonify({"error": "No text provided"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/setup', methods=['POST'])
def setup():
    # This is a placeholder endpoint that Label Studio needs to validate the backend
    return jsonify({"status": "ready"})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_func()
    return jsonify({"status": "shutting down"})

def run_app():
    app.run(host='0.0.0.0', port=5000)

# if __name__ == '__main__':
#     run_app()
