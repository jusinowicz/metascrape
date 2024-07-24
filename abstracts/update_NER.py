"""
Update the NER with user-annotated data from Label Studio
	- Check the configuration file config_abstracts.csv for configuration.
	- Config needs: 
		ncbi_api_key: The NCBI api key
		cache_dir: Where the module stores the abstracts
		query: The query to PubMed, make sure it follows their rules! 
		ls_api_key: API key for Label Studio
		label_studio_url: The local location where it is running
		project_id: Which project id? 
	- This module does not interface directly with Label Studio. HOWEVER
		it does require the latest annotations from a project as a .json
		file, stored in the label_studio_projects directory. This can be 
		easily exported from within Label Studio. 
"""
#libraries
try:
	import os
	import csv
	import sys
	import json
	import glob

	#Spacy NER
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import clean_annotations
except ImportError as e:
	print(f"Failed to import module: {e.name}. Please ensure it is installed.")
	sys.exit(1)
#==============================================================================
def main():
	config_file_path = './config_abstracts.csv'
	try:
		config = load_config(config_file_path)
		#==============================================================================
		# Link to Label Studio to label text
		#==============================================================================
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		model_save_dir = get_config_param(config, 'model_save_dir', required=True)
		latest_labels = get_config_param(config, 'latest_labels', required=True)
		print("Config_abstracts successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	try:
		#Get the list of project files. 
		current_json = {f for f in os.listdir(latest_labels) if f.endswith('.json')}
		current_json = list(current_json)
		#Order by creation to get the newest annotations
		current_json.sort(key=lambda f: os.path.getmtime(os.path.join(latest_labels, f)), reverse=True)
		ll = current_json[0]

		# Load the exported data from Label Studio
		with open(ll, 'r', encoding='utf-8') as file:
    		labeled_data = json.load(file)

		# Filter out entries that have not been annotated by a human: 
		labeled_data = [task for task in labeled_data if 'annotations' in task and task['annotations']]

		# Clean Annotations Function
		cleaned_data = clean_annotations(labeled_data)
