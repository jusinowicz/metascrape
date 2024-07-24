#==============================================================================
# Create or redefine the labeling configuration in a label studio project 
# based on XML and update in the project. 
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

#for label studio interactions: upload_task
import requests  

#the custom modules
sys.path.append(os.path.abspath('./../'))  
from common.config import load_config, get_config_param, ConfigError
#==============================================================================

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
		#==============================================================================
		# If needed: Define the labeling configuration XML and update
		#==============================================================================
		label_config= get_config_param(config, 'label_config_xml', required=True)
		with open(label_config, 'r') as file:
			label_config_xml = file.read()
		# Update the project with the new labeling configuration
		response = requests.patch(
			f'{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}',
			headers={'Authorization': f'Token {API_KEY}', 'Content-Type': 'application/json'},
			json={'label_config': label_config_xml}
		)
		print("Status Code:", response.status_code)
		print("Response Text:", response.text)
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()