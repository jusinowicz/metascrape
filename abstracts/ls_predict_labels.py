#==============================================================================
# Use a NER to predict labels for a Label Studio project
# 	- Check the configuration file config_abstracts.csv for configuration.
#	- Config needs: 
#		ls_api_key
#		label_studio_url
#		project_id
# 	- Label Studio needs to be running. Default location: http://localhost:8080
#	
#==============================================================================
#libraries
import os
import csv
import sys

#For label studio interactions
import requests
from label_studio_sdk import Client  

#the custom modules
sys.path.append(os.path.abspath('./../'))  
from common.config import load_config, get_config_param, ConfigError
#==============================================================================
# Filter out completed tasks
def is_task_completed(task):
    return len(task['annotations']) > 0  # Adjust this condition based on your project's definition of "completed"

def main():
	config_file_path = './config_abstracts.csv'
	try:
		config = load_config(config_file_path)
		#==============================================================================
		# Link to Label Studio to label text
		#==============================================================================
		LS_API_KEY = get_config_param(config, 'ls_api_key', required=True)
		LABEL_STUDIO_URL = get_config_param(config, 'label_studio_url', required=True)
		PROJECT_ID = get_config_param(config, 'project_id', required=True)
		ntasks = get_config_param(config, 'ntasks', required=True)
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	# Check if Label Studio is running
	try:	
		response = requests.get(LABEL_STUDIO_URL)
		if response.status_code != 200:
			print("Label Studio is not running. Please start Label Studio first.")
			sys.exit(1)
	except requests.exceptions.RequestException as e:
		print("Label Studio is not running. Please start Label Studio first.")
		sys.exit(1)

	try:
		# Initialize the Label Studio client
		ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)

		# Get the project
		project = ls.get_project(PROJECT_ID)

		# Fetch tasks from the project
		tasks = project.get_tasks()
		incomplete_tasks = [task for task in tasks if not is_task_completed(task)]

		# Prepare a list to hold the predictions
		predictions = []
		
		# Process the first "ntasks" incomplete tasks
		for task in incomplete_tasks[:ntasks]:
		    text = task['data']['text']  # Adjust this key based on your data format
		    response = requests.post('http://localhost:5000/predict', json={'text': text})
		    predictions_response = response.json()
		    # Prepare predictions in Label Studio format
		    annotations = [{
		        "from_name": "label",
		        "to_name": "text",
		        "type": "labels",
		        "value": {
		            "start": pred['start'],
		            "end": pred['end'],
		            "labels": [pred['label']]
		        }

		    } for pred in predictions_response]
		    # Append the prediction to the list
		    predictions.append({
		        'task': task['id'],
		        'result': annotations, 
		        'model_version': 'custom_web_ner_abs_v381'  # You can set this to track the version of your model
		    })

		# Create predictions in bulk
		project.create_predictions(predictions)