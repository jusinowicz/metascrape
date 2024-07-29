"""
Load text from PDF fulltext files. Export paper sections to Label Studio.
	- Check the configuration file config_fulltext.csv for configuration.
	- Config needs: 
 		pdf_save_dir: Where the PDFs live (i.e. ./../papers/) 
		ls_api_key: API key for Label Studio
		label_studio_url: The local location where it is running
		project_id: Which project id? 
		sections_wanted: Which section to export? Can only do one at a time. 
 	- Label Studio needs to be running. Default location: http://localhost:8080

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
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import extract_text_from_pdf, preprocess_text, identify_sections,upload_task
except ImportError as e:
	print(f"Failed to import module: {e.name}. Please ensure it is installed.")
	sys.exit(1)
#==============================================================================
def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	config_file_path = './config_fulltext.csv'
	try:
		config = load_config(config_file_path)
		pdf_save_dir = get_config_param(config, 'pdf_save_dir', required=True)
		LS_API_KEY = get_config_param(config, 'ls_api_key', required=True)
		LABEL_STUDIO_URL = get_config_param(config, 'label_studio_url', required=True)
		PROJECT_ID = get_config_param(config, 'project_id', required=True)
		sections_wanted = get_config_param(config, 'sections_wanted', required=True)
		#Define a mapping for paper section variations
		section_mapping = {
		    'abstract': 'abstract',
		    'introduction': 'introduction',
		    'background': 'introduction',
		    'methods': 'methods',
		    'materials and methods': 'methods',
		    'methodology': 'methods',
		    'experimental methods': 'methods',
		    'results': 'results',
		    'findings': 'results',
		    'discussion': 'discussion',
		    'conclusion': 'discussion',
		    'summary': 'discussion'
		}
		print("Config_fulltext successfully loaded")
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
		#Upload text from key sections to Label Studio project
		# Keep track of PDFs that have already been processed
		record_file = 'processed_pdfs.txt'

		# Ensure the record file exists - if this is the first run, e.g.
		if not os.path.exists(record_file):
		    with open(record_file, 'w') as f:
		        pass

		# Load processed PDFs from the record file
		with open(record_file, 'r') as f:
		    processed_pdfs = set(f.read().splitlines())

		# Get the list of current PDFs in the directory
		current_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}

		# Identify new PDFs that have not been processed
		new_pdfs = current_pdfs - processed_pdfs

		# Process the new PDFs
		for pdf in new_pdfs:
		    #1. Open the current PDF and extract text
		    pdf_path = pdf_save_dir + pdf
		    pdf_text = extract_text_from_pdf(pdf_path)
		    #2. Do some preprocessing on sentence text
		    sentences = preprocess_text(pdf_text)
		    #3. Put the sentences into their sections. 
		    sections = identify_sections(sentences,section_mapping)
		    #4.Get the required section(s) and upload to Label Studio
		    sections_text = " ".join(sections.get(sections_wanted, [])) 
		    if sections_text: # Check it exists
		        upload_task(sections_text,LABEL_STUDIO_URL, LS_API_KEY, PROJECT_ID)
		    else:
		        print(f"No {sections_wanted} found for article with PMID: {pdf}")
		    #Keep track of processed PDFs
		    processed_pdfs.add(pdf)

		# Update the record file with the new processed PDFs
		with open(record_file, 'w') as f:
		    for pdf in processed_pdfs:
		        f.write(f"{pdf}\n")

	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()