"""
Use DeepDoctection to extract tables from a scentific paper (PDF), convert 
them into Data Frames (pandas), and export tables with treatmens/responses 
that we want.
- Check the configuration file config_tables.csv for configuration.
- Config needs: 
		pdf_save_dir: Where the PDFs live (i.e. ./../papers/) 
		model_load_dir: Where the fitted/updated NER lives
		label_list: The labels that the NER will recognize that you 
			want to be outputted into the formatted table. 
		results_keywords: What are the results keywords that the model has
			been trained to, that we actually want? E.g. biomass, dry weight,
			yield...

More information:
Use the library DeepDoctection to extract tables from a scentific paper and 
convert them into Data Frames (pandas). 

This is meant to be the 2nd step in paper parsing, which goes to the tables to 
to look for response variables and extract the relevant information. This is the 
last step before trying more complex figure-extraction methods. 

Deepdoctection is its own pipeline that is highly customizable. Please check 
out the useful project notebooks for tutorials: 
https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb

Getting Deepdoctection requires some additional installations that may be 
OS/platform-specific. 
Needs: torchvision timm

Once a table is uploaded, the same custom NER built in Spacy is used to look 
for the response variables of interest. 

Current NER: custom_web_ner_abs_v382

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

See extrat_abstract_dat for further dependency installation notes.  
"""
#==============================================================================
#py -3.8

#libraries
try:
	#General usage, visualization
	import cv2
	from pathlib import Path
	from matplotlib import pyplot as plt
	from IPython.core.display import HTML
	from itertools import zip_longest
	import pandas as pd
	import numpy as np 
	import os
	import sys
	import argparse
	import csv
	#Deepdoctection 
	import deepdoctection as dd
	import re
	
	#NER and NLP
	import spacy
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import pdf_try
	
	#the table utilities
	sys.path.append(os.path.abspath('./'))
	import table_utilities as tu
	from table_utilities import split_cardinal
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
	config_file_path = './config_tables.csv'
	try:
		config = load_config(config_file_path)
		pdf_save_dir = get_config_param(config, 'pdf_save_dir', required=True)
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		column_list= get_config_param(config,'column_list',required=True)
		extracted_tables= get_config_param(config,'extracted_tables',required=True)
		print("config_tables successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	try:
		#Load custom NER
		nlp = spacy.load(model_load_dir)
			   
		#DD analyzer #default THRESHOLD_ROWS: 0.4
		#analyzer = dd.get_dd_analyzer(config_overwrite = ["SEGMENTATION.THRESHOLD_ROWS=0.01"])
		analyzer = dd.get_dd_analyzer()

		# # Get the list of current PDFs in the directory
		# new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}

		#Before processing, check if the file exists. If not, create it.
		pdf_list_path = os.path.join(pdf_save_dir, "pdf_list.csv")

		# Create the file if it doesn't exist
		if not os.path.exists(pdf_list_path):
			with open(pdf_list_path, 'w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(["processed_pdfs"])  # Header row
		
		#Read the file to get a set of processed PDF names.			
		processed_pdfs = set()
		with open(pdf_list_path, 'r') as file:
			reader = csv.reader(file)
			next(reader, None)  # Skip header
			for row in reader:
				if row:
					processed_pdfs.add(row[0])

		#Use the processed_pdfs set to skip already-processed files.
		new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf') and f not in processed_pdfs}

		# Process the PDFs
		# Load the column list to structure the final table
		with open(column_list, mode='r') as file:
			column_list = [value for row in csv.reader(file) for value in row]
			
		data = pd.DataFrame(columns = column_list ) #Initialize final table
		index_p = 1
		
		for pdf in new_pdfs:
			#1.Get PDFs and run through dd
			pdf_path = pdf_save_dir + pdf
			print(f"PDF: {pdf} is {index_p} out of {len(new_pdfs)}")
			study_id = pdf_path.lstrip(pdf_save_dir).rstrip('.pdf')
			
			# Pre-check if the PDF can be read
			if not pdf_try(pdf_path):
				print(f"Skipping {pdf} as it is encrypted or unreadable.")
				continue
			try:
				#Initialize the dd analyzer
				df = analyzer.analyze(path=pdf_path)
				df.reset_state()  # This method must be called just before starting the iteration. It is part of the API.
				pages =[] #Get the pages in the PDF
				for doc in df: 
					pages.append(doc)
					
				#2. Cycle through tables in the pages, look for responses, extract them if they are 
				#there:
				table_num = 1
				for pg in pages:
					for tbls in pg.tables:
						t2 = pd.DataFrame(tbls.csv)
						#Replace blank space with NaN
						t2.replace(r'^\s*$', np.nan, regex=True, inplace=True)
						#Drop columns and rows of NaN created by spaces
						t2.dropna(axis=1, how='all', inplace=True)
						t2.dropna(axis=0, how='all', inplace=True)
						t2 = t2.reset_index(drop=True)  
						# Remove letters and special symbols from numbers
						# Remove leading/trailing whitespace from all cells and make all 
						# lowercase
						t2= t2.applymap(tu.clean_numbers)
						# Check the case where there are just a few straggler entries  in 
						# the middle or at the end of a row. Usually means a word connects to 
						# the cell above it.
						t2 = tu.merge_danglers(t2)
						#Check rows with NaN and handle accordingly. This step is designed
						#to help identify headers/subheaders which might only fill a single cell
						for index, row in t2.iterrows():
							if (row.isna().sum() ) >= len(t2.columns)/2+1:
								#Fill NaN with previous cell in row
								row = row.fillna(method='ffill')
								t2.iloc[index,:] = row
						# Apply the classification function to each cell in the DataFrame
						classified_t2 =tu.classify_cells(t2)  
						#Check whether a row is the same type as previous row
						same_type = tu.is_same_type(classified_t2)
						#Use this information to find column headers and parse the table
						#Try to infer which rows are likely to contain headers based on 
						#where the data type across a row changes. If there seem to be 
						if not t2.empty or not same_type.empty:
							#multiple header rows then divide the table into multiple tables. 
							final_tables = tu.organize_tables(t2,same_type)
							#Use the output to grab the correct info from each table and format it and
							#convert it to the write format for output (to match the table format from 
							#the main text, in extract_responses_txt_v2.py)
							final_df = tu.make_final_table(final_tables, study_id, nlp)
							final_df = final_df.reset_index(drop=True)
							print(f"final_df{final_df}")
							data = pd.concat([data, final_df[column_list] ], axis=0 )        # Export DataFrame to a CSV file
							data.to_csv(extracted_tables, index=False)
						#34797549.1_172.pdf
						
						# If successful, add to the processed list
						with open(pdf_list_path, 'a', newline='') as file:
							writer = csv.writer(file)
							writer.writerow([pdf])
						
			except Exception as e:
				print(f"An error occurred: {e}")
				continue
			index_p +=1
						
		#Clean up the table, add SE, link to DOIs
		# Load the DOI mapping file
		doi_mapping = pd.read_csv('all_DOIS.csv')
		doi_mapping['PMID'] = doi_mapping['PMID'].astype(str)
		
		# # Clean the STUDY ID to remove any tags (e.g., ".14")
		#data['STUDY_CLEAN'] = data['STUDY'].astype(str).str.split('.').str[0]
		data['STUDY'] = data['STUDY'].astype(str)
		
		# Merge the DOI information based on the cleaned STUDY ID
		data = data.merge(doi_mapping, left_on='STUDY', right_on='PMID', how='left')
		# Drop the PMID columns
		data = data.drop(columns=['PMID'])
		# #Rename the STUDY_CLEAN column
		# data = data.rename(columns={'STUDY_CLEAN': 'STUDY'})
		# Reorder the columns so that DOI is first and STUDY is second
		columns_order = ['DOI', 'STUDY'] + [col for col in data.columns if col not in ['DOI', 'STUDY']]
		data = data[columns_order]
		
		#When there is also an SE number that was grabbed (this is ideal when it happens) 
		#Apply the split_cardinal function to the CARDINAL column and create the SE column
		data[['CARDINAL', 'SE']] = data['CARDINAL'].apply(split_cardinal).apply(pd.Series)
		
		# Reorder the columns so that SE follows CARDINAL
		columns_order = [col for col in data.columns if col != 'SE']
		se_index = columns_order.index('CARDINAL') + 1
		columns_order.insert(se_index, 'SE')
		
		# Reorder the DataFrame columns
		data = data[columns_order]
		
		data.to_csv(extracted_tables, index=False)
		
	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
	main()