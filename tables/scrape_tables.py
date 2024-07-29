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

    #Deepdoctection 
    import deepdoctection as dd
    import re

    #NER and NLP
    import spacy

    #the custom modules
    sys.path.append(os.path.abspath('./../'))
    from common.config import load_config, get_config_param, ConfigError
        
    #the table utilities
    sys.path.append(os.path.abspath('./'))
    import table_utilities as tu
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
        column_list= get_config_param(config,'column_list',required=True)
        response_table= get_config_param(config,'response_table',required=True)
        print("Config_abstracts successfully loaded")
    except ConfigError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    try:
        # 1. DD analyzer #default THRESHOLD_ROWS: 0.4
        #analyzer = dd.get_dd_analyzer(config_overwrite = ["SEGMENTATION.THRESHOLD_ROWS=0.01"])
        analyzer = dd.get_dd_analyzer()

        # Get the list of current PDFs in the directory
        new_pdfs = {f for f in os.listdir(pdf_save_dir) if f.endswith('.pdf')}

        # Process the PDFs
        # Load the column list to structure the final table
        with open(column_list, mode='r') as file:
            column_list = [value for row in csv.reader(file) for value in row]

        data = pd.DataFrame(columns = column_list ) #Initialize final table
        
        for pdf in new_pdfs:
            #1.Get PDFs and run through dd
            pdf_path = pdf_save_dir + pdf
            print(f"PDF: {pdf} is {index_p} out of {len(new_pdfs)}")
            study_id = pdf_path.lstrip(pdf_save_dir).rstrip('.pdf')
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
                    #multiple header rows then divide the table into multiple tables. 
                    final_tables = tu.organize_tables(t2,same_type)
                    #Use the output to grab the correct info from each table and format it and
                    #convert it to the write format for output (to match the table format from 
                    #the main text, in extract_responses_txt_v2.py)
                    final_df = tu.make_final_table(final_tables, study_id)
                    final_df = final_df.reset_index(drop=True)
                    print(f"final_df{final_df}")
                    data = pd.concat([data, final_df[column_list] ], axis=0 )
                    table_num +=1

        # Export DataFrame to a CSV file
        df.to_csv(extracted_tables, index=False)