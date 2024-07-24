#==============================================================================
# Do a keyword search using the NCBI API. Export the PMIDs with DOIs.
# 	- Check the configuration file config_abstracts.csv for configuration.
#	- Config needs: 
#		ncbi_api_key: The NCBI api key
#		cache_dir: Where the module stores the abstracts
#		doi_save: The output csv for PMIDs and DOIs
# 		query: The query to PubMed, make sure it follows their rules! 
#==============================================================================
#libraries
try:
	import os
	import csv
	import sys

	#for PubMed Fetcher
	from metapub import PubMedFetcher

	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
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
		doi_save = get_config_param(config, 'doi_save', required=True)
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

	    # Create an empty list to store DOIs and PMIDs
		doi_pmid_pairs = []

		# Loop through the articles and extract DOIs and PMIDs
		print("Getting DOIs now")
		for article in articles:
		    doi = article.doi
		    pmid = article.pmid
		    if doi and pmid:
		        doi_pmid_pairs.append((doi, pmid))

        # # Save the DOIs and PMIDs to a CSV file
		with open(doi_save, 'w', newline='') as csvfile:
		    csvwriter = csv.writer(csvfile)
		    csvwriter.writerow(['DOI', 'PMID'])  # Write header
		    for doi, pmid in doi_pmid_pairs:
		        csvwriter.writerow([doi, pmid])
	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()