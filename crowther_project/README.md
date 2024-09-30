# Soil microbiomes and biomass meta analysis project
The goal of this project is to use existing studies to: 
1. Build a database of experimental studies that investigate whether the addition of soil microbiome innocula (defined broadly) impact plant biomass, and 
2. Analyze the extent to which experimental outcomes contain signals of ecosystem type or environmental covariates.

# This is the machine-learning-ASSISTED project
It is not a complete end-to-end pipeline, which is the overarching goal of the project. It uses the code that has been developed to date to: 

1. Download abstracts based on a keyword search (the same used by Averil et al. 2023)
2. A>  Train a ML model (with the help of LabelStudio) to identify key information categories from Abstract text.
2. B> Use ML to extract summary info from the papers, which then allows a human to sort papers by broad categories and prioritize data extraction.
3. Download PDF fulltexts
4. A> Train a second ML model, based on the Abstract model, that can additionaly ID latitude and longitude from the papers. 
4. B> Use PDF tools to extract tables and figures from papers.
5. Use ML to extract tables from papers, which then allows a human to see how many papers actually generated usable information this way. 
6. ASSUMING MOST USEFUL INFORMATION IS IN FIGURES: Use ML to identify figure types, sort them by type, and give them human-readable lables. 
7. A human then has to go through and either by eye or with the help of other tools (i.e. plotdigitizer) extract the data from graphs
8. Enter this data into the master spreadsheet with DOI etc. 

