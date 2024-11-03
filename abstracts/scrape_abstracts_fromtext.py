"""
Use the NER on abstracts to create a preliminary table of variables
- Check the configuration file config_abstracts.csv for configuration.
- Config needs: 
		pdf_save_dir: Where the PDFs live (i.e. ./../papers/) 
		model_load_dir: Where the fitted/updated NER lives
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
	import os, re
	import csv
	import sys
	import json
	import argparse
	import pandas as pd

	#Spacy NER
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import extract_entities, find_entity_groups
	from common.utilities import extract_text_from_pdf, preprocess_text, identify_sections, filter_sentences

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
		pdf_save_dir = get_config_param(config, 'pdf_save_dir', required=True)
		doi_save = get_config_param(config, 'doi_save', required=True)
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		label_list = get_config_param(config, 'label_list', required=True)
		abstract_table= get_config_param(config,'abstract_table',required=True)
		print("Config_abstracts successfully loaded")
		section_mapping = {
			'doi':'doi',	
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
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	#Get the abstracts from the PDFs:
	try:
		#Load custom NER
		nlp = spacy.load(model_load_dir)
		
		# Get the list of current PDFs in the directory
		new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}
		
		# Create an empty list to store articles
		articles = []
		index_p = 1
		for pdf in new_pdfs:
			print(f"PDF: {pdf} is {index_p} out of {len(new_pdfs)}")
			#1. Open the current PDF and extract text
			pdf_path = pdf_save_dir + pdf
			study_id = pdf.rstrip(".pdf")
			pdf_text = extract_text_from_pdf(pdf_path)
			#2. Do some preprocessing on sentence text
			sentences = preprocess_text(pdf_text)
			#print(f"Make it here?{sentences[1:100]}")
			#3. Put the sentences into their sections. 
			sections = identify_sections(sentences,section_mapping)
			# #Filter sentences in the "Results" section
			# abstracts_text = filter_sentences(sections['abstract'], keywords) 
			# print(f"Make it here?{abstracts_text}")
			# #Extract entities from filtered text
			abstracts_text = sections.get('abstract',[])
			doi_text = sections.get('doi', "")
			abstracts_text = " ".join(abstracts_text)
			# # Remove remaining newline characters
			abstracts_text = re.sub(r'\n', ' ', abstracts_text)
			articles.append({'doi':doi_text, 'abstract': abstracts_text})
			index_p += 1
			
	except Exception as e:
		print(f"Something failed with the abstract fetching: {e}")
		
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
			if article:
				# Apply NER to the Abstract to identify treatments and covariates
				abstract_text = article.get('abstract')
				doc, entities = extract_entities(abstract_text,nlp)
				#print(entities)
				row = {"DOI": article.get('doi')}
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