# Fulltext scraping
**This folder contains**: code for downloading full text PDFs from WOS and NCBI, parsing full text PDFs into text blocks by section, figures, and tables, parsing text based on labels defined in Label Studio, training an NER, updating an NER, and attempting to scrape variable TREATMENT and RESPONSE data from the main text which is output as a csv table in output, e.g. /output/extract_from_text1.csv (see config_fulltext.csv to change this). 

## How to use this module
This is a step-by-step example of how the contents of this folder (the module) can be utilized.

### 1.Keyword search and get full texts
The config_fulltext.csv file has a line *query* where the keyword query should be entered. By default, this queries the NCBI data so be sure to follow their search style. YOU WILL NEED AN NCBI KEY. The code accesses the API, for which you will need to go to the NCBI website, set up an account (it's free), and then find the access key under your account profile and enter it into the config file (ncbi_api_key). Then you can run **get_fulltexts.py** which will create a table of DOIs for all of the relevant abstracts it finds (saved by default at ./../output/all_DOIs.csv).

There is an alternative file for Web of Science (WOS) called **get_fulltexts_WOS.R** to try the same search. This is more effective if you are on internet with institutional access. 

### 2. Create a Label Studio Project. 
Even if you have already done this for the Abstracts you should do it a second time here. This is to pull labeled text from sections of the full text that are not likely to appear in abstracts. For example, I have discovered that site Lat/Lon almost never appears in the abstract, yet this can often be an important category of information to extract from a study. For the purpose of the study here, the second Label Studio project is almost identical to the first one created for Abstracts, but contains the two additonal labels LAT and LON. 

As with the abstracts, Label Studio is the core tool used in this project for training NERs. It is free to use and can be downloaded and installed on any system. Follow the general information for Label Studio found in the README in the **abstracts** folder. 

Create a second Label Studio project and name it, and then make sure you keep track of the project ID. Then in the config file, find the *project_id* line and enter it here. This will be different than the ID you created for the Abstracts. 

The file *label_config.xml* again needs to be set up for the labels in your project. The default one for this project is already configured. Feel free to use it, or use it as a template by adding or removing entries. 

The first time that you run this module, run **ls_make_labels.py** which needs the LS_API_KEY, LABEL_STUDIO_URL, PROJECT_ID so be sure these are set. Running this for the first time will create the label structure in your Label Studio project. You should see this once you refresh the project. 

### 3. Add fulltext sections to Label Studio
Next, run **ls_new_sections.py** which will access the downloaded PDFs, parse them into sections, identify the sections you want to use/train the NER on (only using the Methods by default), and finally upload these text blocks to your Label Studio project. 

Now switch over again to the Label Studio project. Again, if you have not annotated before I suggest you check out a brief tutorial on using Label Studio. Now you will annotate the blocks of Methods text that should now appear in your project. As you scroll through a text block, highlight words that you would like the model to recognize as one of your chosen labels. Note, the models only work with 1:1 correspondence, so don't try to label a word with multiple labels. 

### 4. Label, update NER, predict labels, repeat
My recommendation is to annotate at least 20 of these the first time, if you are starting from scratch. By default, starting from scratch actually means that you are training from the NER model trained on Abstracts. So for the most part training should proceed much faster. Train/update the first Methods NER. Then use the trained NER  for labelling by generating predictions, which will appear in label studio. As you continue to label more text you will see the predictions get better and better, and as they get better you will spend less of your own time correcting the labels. 

Run **update_NER.py** which will retrain your NER on the new annotations. 

Then, you can run the updated NER model to generate predictions. Open another terminal or command window and run **host_NER_model.py** (i.e. py -3.8 host_NER_model.py). Then, once the server has been started (you will see output in the terminal), switch to your original terminal and run **ls_predict_labels.py** to generate new predictions. This feeds the fresh text to the NER for predictions. Once it has made predictions, **ls_predict_labels.py** automatically uploads these to your Label Studio project for your convenience. 

The models in *models/* have been trained on 200+ Methods sections in addition to the 800+ that the abstracts NER starts with. Its error rate is fairly low already at this threshold (<10%). 

### 5. Scrape Treatments and Responses
The final step is to generate data with **scrape_responses.py,** which will run through the fulltext, label words, categorize them and attempt to identify response and treatment relationships through analysis of sentence syntax. The code attempts to automatically sort the extracted data into a table and output to /output/extracted_from_text1.csv  

Here is an overview of the table: 
- The final output is a df/CSV with columns STUDY, TREATMENT, RESPONSE, CARDINAL, PERCENTAGE, SENTENCE, ISTABLE. There are separate columns for CARDINAL (a numeric) response) and PERCENTAGE because the NER recognizes them separately. This is useful because it helps determine whether actual units of biomass response are being identified or the ratio of response (percentage). 

- SENTENCE is the sentence that was parsed for the information in the table ISTABLE is a numeric ranking from 0-2 which suggests how likely it is that the the parsed information came from a table that the pdf-parsing grabbed. If the text is grabbed from a table the results are most definitely not to be trusted. So anything that is a 2 should be a hard reject, a 0 is fairly trustoworthy, and a 1 will probably require human assessment. 

- This table is meant to help determine what information is available in the paper and indicate whether further downstream extraction is necessary. 

