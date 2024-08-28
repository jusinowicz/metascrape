"""
Use the NER on Results in fulltext to create a preliminary table of 
treatments and responses.
- Check the configuration file config_fulltext.csv for configuration.
- Config needs: 
		pdf_save_dir: Where the PDFs live (i.e. ./../papers/) 
		model_load_dir: Where the fitted/updated NER lives
		label_list: The labels that the NER will recognize that you 
			want to be outputted into the formatted table. 
		results_keywords: What are the results keywords that the model has
			been trained to, that we actually want? E.g. biomass, dry weight,
			yield...
More information: 
Use NLP and a custom NER model to extract the TREATMENTs and RESPONSEs from 
the text of the Methods and Results sections in scientfic papers. 

This is meant to be the 1st step in paper parsing, trying to glean info from 
the text before trying more complex table-extraction and figure-extraction
methods. 

The final output is a df/CSV with columns STUDY, TREATMENT, RESPONSE, CARDINAL,
PERCENTAGE, SENTENCE, ISTABLE. 
There are separate columns for CARDINAL (a numeric)
response) and PERCENTAGE because the NER recognizes them separately. This is 
useful because it helps determine whether actual units of biomass response are 
being identified or the ratio of response (percentage). 

SENTENCE is the sentence that was parsed for the information in the table 
ISTABLE is a numeric ranking from 0-2 which suggests how likely it is that the
the parsed information came from a table that the pdf-parsing grabbed. In this
case, the results are most definitely not to be trusted. 

This table is meant to help determine what information is available in the paper
and indicate whether further downstream extraction is necessary. 

See extrat_abstract_dat for installation notes.  

This code works fairly well now, but further downstream processing coul be 
implemented to help human eyes interpret and sift through the useful information.
In partiular, removing (or at least flagging) entries that appear to be numbers 
grabbed from summary statistics (e.g. p-values, F-values, AIC, etc.). This seems 
to happen frequently. 
"""
#libraries
try:
	import os
	import csv
	import sys
	import json
	import argparse
	import pandas as pd
	import re

	#Post-NER processing
	from collections import defaultdict

	#Spacy NER
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import extract_text_from_pdf, preprocess_text, identify_sections, filter_sentences, create_table, extract_entities
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
	config_file_path = './config_fulltext.csv'
	try:
		config = load_config(config_file_path)
		pdf_save_dir = get_config_param(config, 'pdf_save_dir', required=True)
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		results_keywords= get_config_param(config,'results_keywords',required=True)
		response_table= get_config_param(config,'response_table',required=True)
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
		print("Config_abstracts successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	try:
		#Load custom NER
		nlp = spacy.load(model_load_dir)

		# Load the results keywords
		with open(results_keywords, mode='r') as file:
			keywords = [value for row in csv.reader(file) for value in row]

		# Get the list of current PDFs in the directory
		new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}

		data = []
		index_p = 1
		# Process the new PDFs
		for pdf in new_pdfs:
			print(f"PDF: {pdf} is {index_p} out of {len(new_pdfs)}")
			#1. Open the current PDF and extract text
			pdf_path = pdf_save_dir + pdf
			study_id = pdf.rstrip(".pdf")
			pdf_text = extract_text_from_pdf(pdf_path)
			#2. Do some preprocessing on sentence text
			sentences = preprocess_text(pdf_text)
			#3. Put the sentences into their sections. 
			sections = identify_sections(sentences,section_mapping)
			#Filter sentences in the "Results" section
			results_text = filter_sentences(sections["results"], keywords) 
			#Extract entities from filtered text
			results_text = " ".join(results_text)
			# Remove remaining newline characters
			results_text = re.sub(r'\n', ' ', results_text)
			results_doc, results_entities = extract_entities(results_text, nlp)
			table = create_table(results_doc, results_entities, study_id)
			data.append(table)
			index_p +=1

		flattened_data = [item for sublist in data for item in sublist]
		df = pd.DataFrame(flattened_data)

		# Export DataFrame to a CSV file
		df.to_csv(response_table, index=False)


	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()