"""
Get fulltexts as PDF using the NCBI API. 
	- Check the configuration file config_fulltext.csv for configuration.
	- Config needs: 
		ncbi_api_key: The NCBI api key
		cache_dir: Where the module stores the abstracts
 		query: The query to PubMed, make sure it follows their rules! 
-Note: See the R file (probably in this directory) get_fulltexts_WOS.R
	for a solution that will work consistently. It will download more 
	full texts but to get the full list you may still need some kind of 
	institutional access. 
"""
#libraries
try:
	import os
	import csv
	import sys
	import argparse
	import pandas as pd

	#for PubMed Fetcher
	from metapub import PubMedFetcher
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import get_full_text
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
		NCBI_API_KEY = get_config_param(config, 'ncbi_api_key', required=True)
		os.environ['NCBI_API_KEY'] = NCBI_API_KEY
		cache_dir = get_config_param(config, 'cache_dir', required=True)
		query = get_config_param(config, 'query', required=True)
		print("Config_abstracts successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

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

	    #==============================================================================
		#Try to get full text PDFs from NCBI
		#==============================================================================
		# Directory to save the full text articles
		save_directory = './../papers/'

		# Fetch and save full text articles
		get_full_text(articles, save_directory)

	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
