#==============================================================================
# Do a keyword search using the NCBI API. Export the abstracts to Label Studio.
# 	- Check the configuration file config_abstracts.csv for configuration.
#	- Config needs: 
#		ncbi_api_key: The NCBI api key
#		cache_dir: Where the module stores the abstracts
# 		query: The query to PubMed, make sure it follows their rules! 
#		ls_api_key: API key for Label Studio
#		label_studio_url: The local location where it is running
#		project_id: Which project id? 
# 	- Label Studio needs to be running. Default location: http://localhost:8080
#
#==============================================================================
#libraries
try:
	import os
	import csv
	import sys
	
	#for PubMed Fetcher
	from metapub import PubMedFetcher
	
	#For label studio interactions
	import requests
	from label_studio_sdk import Client
	import json
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import upload_task
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
		NCBI_API_KEY = get_config_param(config, 'ncbi_api_key', required=True)
		os.environ['NCBI_API_KEY'] = NCBI_API_KEY
		cache_dir = get_config_param(config, 'cache_dir', required=True)
		query = get_config_param(config, 'query', required=True)
		LS_API_KEY = get_config_param(config, 'ls_api_key', required=True)
		LABEL_STUDIO_URL = get_config_param(config, 'label_studio_url', required=True)
		PROJECT_ID = get_config_param(config, 'project_id', required=True)
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

	try:
		#Initialize the api
		fetcher = PubMedFetcher(cachedir='./../papers/')

		# Use the fetcher to get PMIDs for the query
		print("Querying the PubMed database")
		pmids = fetcher.pmids_for_query(query,retmax = 10000)

		# Create an empty list to store articles
		articles = []

		print("Getting articles, this can take some time")
		# Get the information for each article: 
		for pmid in pmids:
		    article = fetcher.article_by_pmid(pmid)
		    articles.append(article)

	    # Loop through each abstract
	    for a in articles:
			# Upload abstracts
			abstract = a.abstract
			if abstract: # Check if the article has an abstract
				upload_task(abstract, PROJECT_ID)
			else:
				print(f"No abstract found for article with PMID: {article.pmid}")

	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()