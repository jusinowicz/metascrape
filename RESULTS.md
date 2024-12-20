# Results
Each subsection of the project ([abstracts](/abstracts/README.md), [fulltext](/fulltext/README.md), [tables](/tables/README.md), [figures](figures/README.md)) provides useful output and insight into the broader body of literature generated by a keyword search. I have tried to design tools within each subsection that could be adapted for other explorations either on new bodies of literature, or within the same literature search. Here, I have tried to provide some insight into the way that these tools can be utilized by discussing key output in each subsection. 

There is a folder, [analysis_in_R](/analysis_in_R/) which contains R scripts to perform analyses and generate summaries on some of the output. This code shoud all run, but it should be considered more as suggestive sketches of what can be done, as opposed to final, clean code that creates final-draft quality figures or tables. 

## Abstracts
The primary data object output in the [abstracts](/abstracts/) subsection is the table in [output](/output/) named [abstract_parsing1.csv](/output/abstract_parsing1.csv) (by default). It contains the columns DOI, TREATMENT, INOCTYPE, RESPONSE, SOILTYPE, FIELDGREENHOUSE, LANDUSE, ECOTYPE, ECOREGION, and LOCATION by default. These are labels that the NER model (Named Entity Recognition, a type of deep learning model in machine learning) has been trained to recognize (see the [abstracts README](/abstracts/README.md)). They are the categories created in the original meta analysis database (from Averil et al. 2022). Each row corresponds to one of the articles identified from the keyword search. Each entry is just a list of text showing all of the words or phrases that the NER model identified with the corresponding label.

The R code in [abstract_table_summary.R](/analysis_in_R/abstract_table_summary.R) contains several ways to parse and sort through this table. For starters, I have tried to create lists of keyword identifies that can sort papers into three broad subject areas in which soil microbiome work seems to primarily be done: agriculture, disturbance (e.g. soil remediation, mine reclemation), and general biology (which includes ecology). The studies can be further subdivided from there, which I have attempted to do for the general biology papers. This graphic is a summary of these paper groupings: 

![alt text](/output/paper_summary.jpg)

(Note: This image also exists as a dynamic html object of the same name that you can interact with.)

Additionally, a number of bar plots and word clouds exist in the output folder which can be browsed. However, these can take some fine-tuning to generate useful insight, and the ones that currently exist in this directory are experimental. 

Categorizing the papers in this way is useful for three main reasons. First, it allows us to parse the classic "rejection" logic of meta analyses. The figure above is a map for the criteria we might choose to retain or reject papers based on the goals of the study. For example, we might choose to reject all agricultural and disturbance papers as not being in line with the current objects. 

Whether we choose to reject or retain papers is also useful for allocating work effort and cross-referencing in the later subsections. Especially once it comes to extracting data from figures, this becomes a process that still requires human eyes to complete and so we might not wish to extract data from all 600+ studies.

If (or when) we choose to retain these papers, these general categories are also useful because they become variables in our statistical models. Especially at the broadest level, we might find these as useful random/fixed effects that help explain broad categories of observed biomass responses. 

## Fulltext
The primary data object output in the [fulltext](/fulltext/) subsection is the table in [output](/output/) named [extract_from_text1.csv](/output/extract_from_text1.csv) (by default). It contains the columns STUDY, TREATMENT, RESPONSE, CARDINAL, PERCENTAGE, SENTENCE, ISTABLE. 

The modules first take the full text from a paper and use the NER to label the text. Then, using another layer of ML models, syntactical parsing, and some other tricks, the modules attempt to find text that relates TREATMENTs to specific RESPONSEs (e.g. biomass, dry weight) and includes numerical (either CARDINAL or PERCENTAGE) details. The modules then also record the sentence that the data were supposedly found in, and a ranking in ISTABLE from 0-3 that relate to the quality of the data (0 is the best, 3 is the worst). The ranking in ISTABLE is meant to clarify whether these are actually text from sentences or unintentionally picked up from tables or other structures embedded in the document. This is a frequent problem with PDFs, but the ISTABLE ranking helps. In short, I would not trust any results where ISTABLE > 0.

The R code in [fulltext_table_summary.R](/analysis_in_R/fulltext_table_summary.R) provides some quick summaries about the number of succesful data reads from the text, picks out the high-quality table data, then references the data extracted from the abstracts to combine the results. Importantly, it also create the full output that combines variables from across abstracts, methods, and results in [fulltext_abstracts_comb.csv](/output/fulltext_abstracts_comb.csv). The columns of this object are: 

 [1] "STUDY"           "TREATMENT"       "RESPONSE"        "CARDINAL"       
 [5] "PERCENTAGE"      "SENTENCE"        "ISTABLE"         "DOI"
 [9] "LAT"             "LON"             "INOCTYPE"        "SOILTYPE"
[13] "FIELDGREENHOUSE" "LANDUSE"         "ECOTYPE"         "ECOREGION"
[17] "LOCATION"

The naming of these variables corresponds to the labels created in Label Studio for the ML models that are extracting information from the text. Each of these corresponds directly to one of the columns in the original meta analysis database from Averil et al.

## Tables
The results from this module are directly analogous to those created from the fulltext. The goal is to extract actual tables from the papers that include the information we're looking for. Scraping tables themselves seems like a more effective approach, in part because more papers report results in tables as opposed to text, tables are more standard in their format, and especially because tables are more likely to provides the SE as well as the effect size.

The main data object output direcly by the modules in the [tables](/tables/) subsection is the table in [output](/output/) named [extract_from_tables1.csv](/output/extract_from_tables1.csv) (by default).

The R code in [table_summary.R](/analysis_in_R/table_summary.R) provides some quick summaries about the number of succesful data reads from the text, picks out the high-quality table data, then references the data extracted from the abstracts to combine the results. Importantly, it also create the full output that combines variables from across abstracts, methods, and results in [table_full_out.csv](/output/table_full_out.csv). The columns of this object are: 

 [1] "STUDY"           "TREATMENT"       "RESPONSE"        "CARDINAL"       
 [5] "PERCENTAGE"      "SENTENCE"        "ISTABLE"         "DOI"
 [9] "LAT"             "LON"             "INOCTYPE"        "SOILTYPE"
[13] "FIELDGREENHOUSE" "LANDUSE"         "ECOTYPE"         "ECOREGION"
[17] "LOCATION"

The naming of these variables corresponds to the labels created in Label Studio for the ML models that are extracting information from the text. Each of these corresponds directly to one of the columns in the original meta analysis database from Averil et al.