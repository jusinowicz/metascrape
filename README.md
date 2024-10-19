# Soil microbiomes and biomass meta analysis project
The goal of this project is to use existing studies to: 
1. Build a database of experimental studies that investigate whether the addition of soil microbiome innocula (defined broadly) impact plant biomass, and 
2. Analyze the extent to which experimental outcomes contain signals of ecosystem type or environmental covariates.

This repository is structured to reflect the workflow of the project. Each folder contains its own Readme with more details.  

## Main workflow 
**Note**: Each folder contains its own README, which contains detailed information and tutorials on running the modules. The suggested order of modules and READMEs is: 
1. abstracs
2. fulltext
3. tables
4. figures. 

**abstracts**: This folder contains the initial database-building arm of the project. It is built on a series of python scripts that scrape abstracts from [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6821292/), train an NER utilizing [spaCy](https://spacy.io/usage/spacy-101) and [label-studio](https://labelstud.io/), provides the pipeline for initial NER training and updating, and creates a table of useful keywords from the categories used to train the NER. 

**figures**: Once the fulltexts have been downloaded and parsed, identify figures and sort them based on their type (e.g. scatter plot, bar plot, etc.). Ideally, code to extract the relevant data from each figure would exist here. However, this is the most experimental folder, as training ML models to reliably extract data from scatter plots and bar plots is challenging. 

**fulltext**: The main code to download and parse PDFs, identify and separate text from tables and figures, and automate an initial extractions of TREATMENT and RESPONSE data from the text into tables for the analysis. The modules to train/update a second NER based on Methods sections exists here. The purpose of this second NER is largely to add in Latitude and Longitude categories and automate site extraction information from the Methods of each paper. 

**tables**: Once the fulltexts have been downloaded and parsed, identify tables, identify tables that relate to TREATMENT and RESPONSE data, and extract/convert them to the meta analysis format. 

 **Note**: Each folder contains a config_*text*.csv, where *text* is the pipeline folder. The config file controls file and folder locations, names and certain other configuration components

## Other folders
**analysis_in_R**: Various R code to help parse and analyse output from scraping. This folder contains the R scripts to analyze the database, assuming it is available as several csv files. It creates output ranging from figures of variable importance (RandomForests), covariate significance and effect size (mixed-effect models), and various saved models. 


**common**: Shared modules used across the project

**label_studio_projects**: These are the annotated files for Label Studio, and that are used to train the NERs

**models**: The fitted NER models 

**output**: The top-level folder for all of the various output generated by the scraping and the R analysis. 


Each project folder contains its own README which further details.


## Installation notes.
Unless otherwise noted, all of the python code here runs on python 3.8. This was the most current version that seemed able to run all of the NER training modules. If you have never run python before, I suggest looking into using either py to choose and run a specific python environment (used often throughout the helps and readmes) or using anaconda to set up and run a custom environment. 

py -3.8 -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=myenv --display-name "Python 3.8.10 (myenv)"
