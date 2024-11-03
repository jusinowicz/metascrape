"""
If the PMIDs are not known, extract DOIs from text and link them to file names.
	- Check the configuration file config_abstracts.csv for configuration.
	- Config needs: 
		pdf_save_dir: Where the PDFs live (i.e. ./../papers/) 
		model_load_dir: Where the fitted/updated NER lives
		doi_save: The output csv for PMIDs and DOIs
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
	import nltk
	nltk.download('punkt_tab')
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import extract_text_from_pdf, preprocess_text, identify_sections, filter_sentences
except ImportError as e:
	print(f"Failed to import module: {e.name}. Please ensure it is installed.")
	sys.exit(1)
#==============================================================================
def main():
	# Parse command-line arguments
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	config_file_path = './config_abstracts.csv'
	try:
		config = load_config(config_file_path)
		pdf_save_dir = get_config_param(config, 'pdf_save_dir', required=True)
		doi_save = get_config_param(config, 'doi_save', required=True)
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
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

	try:
		#Load custom NER
		nlp = spacy.load(model_load_dir)
		
		# Get the list of current PDFs in the directory
		new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}
		
		# Create an empty list to store DOIs and PMIDs
		doi_pmid_pairs = []
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
			#print(f"Make it here? {sections.get('doi')}")
			#Get the doi
			doi_text = sections.get('doi')
			if doi_text and study_id:
				doi_pmid_pairs.append((doi_text, study_id))
			
			index_p +=1
		        
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