"""
Use the NER on abstracts to create a preliminary table of variables
- Check the configuration file config_abstracts.csv for configuration.
- Config needs: 
		ncbi_api_key: The NCBI api key
		cache_dir: Where the module stores the abstracts
		query: The query to PubMed, make sure it follows their rules! 
		model_save_dir: Where to save the fitted/updated NER
		label_list: The labels that the NER will recognize that you 
			want to be outputted into the formatted table. 

More information: 
The code will cycle through a list of abstracts, extract the pertanent 
information, and either create or add the information to a spreadsheet.
The custom NER model extracts information into the following categories: 

TREATMENT: Could be any number of inoculation, nutrient, environmental
			 experimental protocols.  
INOCTYPE: The type of inoculant. Could be species, group (e.g. AMF), or
			more generic (e.g. soil biota)
RESPONSE: Should be either biomass or yield 
SOILTYPE: The type of soil
FIELDGREENHOUSE: Is the experiment performed in a greenhouse or field
LANDUSE: For experiments where the context is a history of land use, 
			e.g. agriculture, urban, disturbance (pollution, mining) 
ECOTYPE: Could be true ecotype (e.g. wetlands, grasslands, etc.) or a 
			single species in the case of ag studies (e.g. wheat)
ECOREGION: Reserved for very broad categories. 
LOCATION: If given, a geographic location for experiment


Each entry in the spreadsheet will actually be a list of possible values.
For example, TREATMENT could be a list of "fertilizer, combined inoculation,
sterilized soil, AMF..." The resulting spreadsheet is meant to be its 
own kind of database that a researcher could use to check whether each 
article (by its DOI) is likely to contain the information they need. 
The next step in the processing pipeline would be code to use something 
like regular expression matching to identify these studies from the table
created here. 

Installation notes: 
This package needs to be installed for NLP: 
py -3.8-m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
# py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz

Python 3.8 was the newest version I successfully installed this on!
It seemed possible maybe with 3.11,3.10.3.9 but very finicky to set up. 

My recommendation is to install this first and let it install its own 
version of spacy and dependencies (something with pydantic versions seems
to be the problem).

The package en_core_sci_md also requires that C++ is installed on your system, so visual studio build
tools on Windows.

py -m pip install PyPDF2 pdfplumber tabula-py jpype1 PyMuPDF Pillow nltk
For spacy:
py -m spacy download en_core_web_sm
For NER: 

py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
py -3.8 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_craft_md-0.5.0.tar.gz

"""
#libraries
try:
	import os
	import csv
	import sys
	import json
	import argparse
	import pandas as pd

	#for PubMed Fetcher
	from metapub import PubMedFetcher

	#Spacy NER
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import extract_entities, find_entity_groups
except ImportError as e:
	print(f"Failed to import module: {e.name}. Please ensure it is installed.")
	sys.exit(1)
#==============================================================================
def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--new', action='store_true', help='Create a new NER instead of retraining an existing one.')
	args = parser.parse_args()
	#Open config file
	config_file_path = './config_abstracts.csv'
	try:
		config = load_config(config_file_path)
		NCBI_API_KEY = get_config_param(config, 'ncbi_api_key', required=True)
		os.environ['NCBI_API_KEY'] = NCBI_API_KEY
		cache_dir = get_config_param(config, 'cache_dir', required=True)
		query = get_config_param(config, 'query', required=True)
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		label_list = get_config_param(config, 'label_list', required=True)
		abstract_table= get_config_param(config,'abstract_table',required=True)
		print("Config_abstracts successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	#Get the abstracts based on keyword search:
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

	except Exception as e:
		print(f"Something failed with the PubMedFetcher: {e}")

	#Scrape the loaded abstracts
	try:
		#Load custom NER
		nlp = spacy.load(model_load_dir)

		# Load the labels
		with open(label_list, mode='r') as file:
			labels_use = [value for row in csv.reader(file) for value in row]

		# Create a DataFrame with columns for each label category, plus the DOI
		columns = ["DOI"]+ labels_use

		# Initialize a list to collect rows
		rows = []
		#Cycle through the articles, get the information, and add to the table
		for article in articles:
			# Apply NER to the Abstract to identify treatments and covariates
			abstract_text = article.abstract
			doc, entities = extract_entities(abstract_text,nlp)
			#print(entities)
			row = {"DOI": article.doi}
			#Cycle through the labels
			for label in labels_use:
				#Find the groups
				entity_matches = find_entity_groups(doc, entities, label)
				row[label] = "; ".join(entity_matches)  # Join matches with a separator, e.g., '; '
			# Collect the row
			rows.append(row)

		#Create and save the df
		df = pd.DataFrame(rows, columns=columns)
		df.to_csv(abstract_table, index=False)

	except Exception as e:
		print(f"Something failed while scraping abstracts: {e}")

if __name__ == "__main__":
    main()