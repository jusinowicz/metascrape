"""
Custom functions that are only used for getting and processing the 
tables. 

"""

#==============================================================================
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

#Deepdoctection 
import deepdoctection as dd
import re

#NER and NLP
import spacy

#the custom modules
sys.path.append(os.path.abspath('./../'))
from common.utilities import extract_entities

#==============================================================================
# Custom functions: 
#==============================================================================
# Remove letters and special symbols from numbers
# Remove leading/trailing whitespace from all cells and make all 
# lowercase
def clean_numbers(cell):
    if isinstance(cell, str):
        # Remove leading and trailing whitespace
        cell = cell.strip().lower()
        # Remove special characters
        cell = re.sub(r'[^\w\s.]', '', cell)
        # Remove non-numeric characters after numbers
        cell = re.sub(r'(\d+(\.\d+)?)(\s*[a-zA-Z]+)?', r'\1', cell)
        # Convert to float if numeric
        if not pd.isna(cell):
            try:
                cell = float(cell)
            except ValueError:
                pass  # If conversion fails, cell remains unchanged    
    return cell

#Sometimes when headers have multiple lines of text the second+ lines
#create a new row. This is a way to find and merge cells like this. 
def merge_danglers(table):
    rows_to_drop = []
    for index, row in table.iloc[1:].iterrows():
        #print(f"{index}, {row}")
        if (row.isna().sum() ) >= len(table.columns)/2+1:
                #print(f"Na Sum, {row.isna().sum()}")
                if not row[2:len(table.columns)+1].isna().all():
                    # Iterate through each column
                    #print(f"Not, {row[2:len(table.columns)+1].isna().all()}")
                    for col in table.columns:
                        # If the entry is not NaN, concatenate it to the previous row
                        if not pd.isna(row[col]):
                            table.at[index - 1, col] = str(table.at[index - 1, col]) + " " + str(row[col]) if not pd.isna(table.at[index - 1, col]) else row[col]
                    # Add the current row index to the list of rows to drop
                    rows_to_drop.append(index)
    # Drop the flagged rows
    table.drop(rows_to_drop, inplace=True)
    # Reset index
    table.reset_index(drop=True, inplace=True)
    return table

#This one is experimental and cannot yet be reliably used.
#Sometimes there is a situation where a row label has a dangler.
#But as of yet, I cannot figure out a way that catches danglers
#while avoiding treating a new subheading as a dangler. 
def merge_danglers_two (table, same_type):
    rows_to_drop = []
    for index, row in table.iloc[1:len(table)].iterrows():
        # Check if all entries in same_type except the first column are False
        if same_type.iloc[index,1:].apply(lambda x: x is False).sum() >= len(table.columns)/2+1:
            #Check if the current row is also all strings
            if table.iloc[index,1:].apply(lambda x: isinstance(x, str)).all():
                #Check if the next row is also (mostly) False
                if same_type.iloc[index+1,1:].apply(lambda x: x is False).sum() >= len(table.columns)/2+1:            
                    #Then we assume the first column was a dangler and this needs
                    #to be corrected:
                    table.at[index - 1, 0] = str(table.at[index - 1, 0])  + " " + str(table.at[index, 0])
                    # Add the current row index to the list of rows to drop
                    rows_to_drop.append(index)
    # Drop the flagged rows
    table.drop(rows_to_drop, inplace=True)
    # Reset index
    table.reset_index(drop=True, inplace=True)
    return table

#Try to detect and combine header rows
def classify_cells(table):
    classifications = pd.DataFrame(index=table.index, columns=table.columns)
    for row in range(0,len(table)):
        for col in range(0, len(table.columns)):
            cell = table.iloc[row, col]
            if pd.isna(cell):
                classifications.iloc[row, col] = 'NaN'
            elif isinstance(cell, str):
                classifications.iloc[row, col] = 'string'
            elif isinstance(cell, (int, float)):
                classifications.iloc[row, col] = 'numeric'
            else:
                classifications.iloc[row, col] = 'other'
    return classifications

# Function to check if cell type is same as previous cell in the column
def is_same_type(table):
    results = pd.DataFrame(index=table.index, columns=table.columns, dtype=bool)
    for col in range(0, len(table.columns)):
        for row in range(1, len(table)):  # Start from the second row
            current_type = (table.iloc[row, col])
            previous_type = (table.iloc[row-1, col])
            results.iloc[row, col] = current_type == previous_type
    return results


#Use this information to find column headers and parse the table
#Try to infer which rows are likely to contain headers based on 
#where the data type across a row changes. If there seem to be 
#multiple header rows then divide the table into multiple tables. 

def organize_tables(table, same_type):
    final_tables = []
    #Determine which case we're dealing with:
    #Number of (sub)header rows:
    nheaders = len(same_type)-same_type.sum()
    whichrow = same_type.shape[1]-same_type.sum(axis=1)
    if len(nheaders) > 1:
        nheaders = nheaders.iloc[1]
    else:
        nheaders = nheaders.iloc[0]
    #Case 0: First row is headers, single row. Could be two 
    #ways this appears, with either no FALSE row or FALSE row is 
    #in row[1].
    if nheaders == 0:
        ft = pd.DataFrame(table.iloc[1:,:])
        ft = ft.rename(columns = (table.iloc[0,:]))
        final_tables.append(ft)
    if nheaders == 1:
        if whichrow[1]>0:
            ft = pd.DataFrame(table.iloc[1:,:])
            ft = ft.rename(columns = (table.iloc[0,:]))
            final_tables.append(ft)
        #Case 1: Multiple headers at the top of table
        else:
            # Find the index where the type switch occurs (row with False values)
            # Ignore the first row and first column, then find the first occurrence of False
            header_index = same_type.iloc[1:, 1:].eq(False).idxmax().max() 
            # Concatenate rows 0 to header_index to create new headers
            header_rows = table.iloc[:header_index]
            new_headers = header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0)
            # Create the new DataFrame with the remaining rows and the new headers
            data_rows = table.iloc[header_index + 1:]
            ft = pd.DataFrame(data_rows.values, columns=new_headers)
            final_tables.append(ft)
    #Case 2: Multiple headers and sub-headers within table 
    elif nheaders > 1:
        #Assume that the first block of headers includes both the main overall 
        #headers, as well as the first row of subheaders: 
        header_index = same_type.iloc[1:, 1:].eq(False).idxmax().max()
        # Concatenate rows 0 to header_index-1 to exclude first sub-header and
        # create new main headers
        header_rows = table.iloc[:header_index-1]
        new_headers = header_rows.apply(lambda x: ' '.join(filter(pd.notna, x.astype(str))), axis=0) 
        # Iterate through the table to find additional sub-headers and divide the table into sub-sections
        current_index = header_index-1
        if current_index+2 < len(table):
            while current_index < len(table):
                next_index = same_type.iloc[current_index+2:, 1:].eq(False).idxmax().max()
                if (next_index+2)>=len(table):
                    next_index = len(table)
                #Create sub-headers
                sub_headers = table.iloc[current_index,:]
                # Combine main headers with sub headers
                combined_headers = [f"{nh} {sh}" for nh, sh in zip_longest(new_headers, sub_headers, fillvalue="")]
                # Extract the sub-table
                sub_table = table.iloc[current_index+1:next_index]
                # Create the final DataFrame for this section
                ft = pd.DataFrame(sub_table.values, columns=combined_headers)
                # Add to the list of final DataFrames
                final_tables.append(ft)
                # Move to the next section
                current_index = next_index
    return final_tables

#Approach 2: This seems to be cleaner and simpler. Just take each header as its own 
#sentence, run it through the NER, and identify whether it contains the response 
#variables.  

#Turn headers into sentences: 
def headers_to_sentences(table):
    sentences = []
    # Iterate through each column
    for col in range(1, table.shape[1]):
        treatment = table.columns[0] 
        response_variable = table.columns[col]
        sentence = (  f"The response variable is {response_variable}, end of sentence.")
        sentences.append(sentence)
    return sentences

#Second, run the header-sentences through the NER
def find_response_cols(table,nlp):
    sentences_with_treatment = [] 
    sent_index = []
    treat_name = []
    sent_now = 1 #We've skipped first column, assume it's treatments
    #Get the response variables from the table by turning them
    #into sentences and passing through the NER
    header_sentence = headers_to_sentences(table)
    header_sentence = " ".join(header_sentence)                                                      
    doc,entities = extract_entities(header_sentence, nlp)
    for sent in doc.sents:
        # Check each entity in the doc
        for ent in doc.ents:
            # Check if the entity is within the sentence boundaries and has the label 'TREATMENT'
            if ent.start >= sent.start and ent.end <= sent.end and ent.label_ == "RESPONSE":
                sentences_with_treatment.append(sent.text)
                sent_index.append(sent_now)
                treat_name.append(ent.text)
                break  # Once we find a TREATMENT entity in the sentence, we can move to the next sentence
        sent_now +=1
    return sent_index, sentences_with_treatment, treat_name

#Use the output to grab the correct info from each table and format it and
#convert it to the write format for output (to match the table format from 
#the main text, in extract_responses_txt_v2.py)

def make_final_table(final_tables, study_id, nlp): 
    column_list = ['STUDY', 'TREATMENT','RESPONSE','CARDINAL','PERCENTAGE','SENTENCE', 'ISTABLE']
    final_df = pd.DataFrame(columns = column_list )
    row_index = 0
    for t1 in final_tables:
        r_index, r_sent, r_name = find_response_cols(t1,nlp)
        #If the response was not found, skip it 
        if not r_index:
            continue
        else:
            #new_rows = t1.iloc[:, r_index + [(max(r_index)+1)]  ] 
            new_rows = t1.iloc[:, [0] + r_index  ].copy() 
            new_rows.columns.values[0] = 'TREATMENT'
            # Melt the DataFrame so that column names become SENTENCE
            nr_melted = pd.melt(new_rows, id_vars=['TREATMENT'], var_name='SENTENCE', value_name='CARDINAL')
            #Add the standardized (i.e. NER label) name of the response
            # Repeat the labels to match the length of nr_melted
            r_name_long = pd.Series(r_name).repeat(len(nr_melted) // len(pd.Series(r_name)) + 1)[:len(nr_melted)]
            nr_melted['RESPONSE'] = r_name_long.values
            #Add the remaining columns: study id, percentage, istable:
            nr_melted['STUDY'] = study_id
            nr_melted['PERCENTAGE'] =''
            nr_melted['ISTABLE'] = 99 #This obviously came from a table
            final_df=pd.concat([final_df, nr_melted[column_list] ], axis=0 )
        #final_df=final_df.reset_index()        
    return final_df
