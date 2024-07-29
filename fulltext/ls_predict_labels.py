"""
Use a NER to predict labels for a Label Studio project
	- Check the configuration file config_abstracts.csv for configuration.
	- Config needs: 
		ls_api_key: API key for Label Studio
		label_studio_url: The local location where it is running
		project_id: Which project id? 
		ntasks: How many tasks to predict over? 
	- Label Studio needs to be running. Default location: http://localhost:8080
	- This module will run host_NER_model to start the NER as a service 
	  at localhost:5000. The process is terminated before the program ends. 
		In Windows: netstat -ano | findstr 5000
"""
#libraries
try:
	import os
	import csv
	import sys
	import argparse
	
	#For label studio interactions
	import requests
	from label_studio_sdk import Client
	import json  
	
	#To start the NER on localhost from within this module:
	import time
	from multiprocessing import Process
	from host_NER_model import run_app
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
except ImportError as e:
	print(f"Failed to import module: {e.name}. Please ensure it is installed.")
	sys.exit(1)
#==============================================================================
# Filter out completed tasks
def is_task_completed(task):
    return len(task['annotations']) > 0  # Adjust this condition based on your project's definition of "completed"

#Main
def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	config_file_path = './config_fulltext.csv'
	try:
		config = load_config(config_file_path)
		LS_API_KEY = get_config_param(config, 'ls_api_key', required=True)
		LABEL_STUDIO_URL = get_config_param(config, 'label_studio_url', required=True)
		PROJECT_ID = get_config_param(config, 'project_id', required=True)
		ntasks = get_config_param(config, 'ntasks', required=True)
		ntasks = int(ntasks)
		print("Config_abstracts successfully loaded")
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

	#If all has gone right up to now then start the Flask app in a separate process
	flask_process = Process(target=run_app)
	flask_process.start()
	print(f"The NER is being started at http://localhost:5000")
	# Wait for the server to start
	time.sleep(15)

	try:
		# Initialize the Label Studio client
		ls = Client(url=LABEL_STUDIO_URL, api_key=LS_API_KEY)

		# Get the project
		project = ls.get_project(PROJECT_ID)

		# Fetch tasks from the project
		tasks = project.get_tasks()
		incomplete_tasks = [task for task in tasks if not is_task_completed(task)]

		# Prepare a list to hold the predictions
		predictions = []
		
		# Process the first "ntasks" incomplete tasks
		print(f"NER running, generating predictions = {ntasks}")
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
	finally:
		print("Done! Shutting down NER model")
		# Shutdown the Flask app
		flask_process.terminate()

if __name__ == "__main__":
    main()