# Abstract scraping
**This folder contains**: code for WOS/NCBIkeyword search, downloading abstracts, parsing text based on labels defined in Label Studio, training an NER, updating an NER, and scraping abstracts for text which is output as a csv table in output, e.g. abstract_parsing1.csv (see config_abstrac.csv to change this). 

## How to use this module
This is a step-by-step example of how the contents of this folder (the module) can be utilized.

### 1. Keyword search
The config_abstracts.csv file has a line *query* where the keyword query should be entered. By default, this queries the NCBI data so be sure to follow their search style. YOU WILL NEED AN NCBI KEY. The code accesses the API, for which you will need to go to the NCBI website, set up an account (it's free), and then find the access key under your account profile and enter it into the config file (ncbi_api_key). Then you can run **abstract_doi.py** which will create a table of DOIs for all of the relevant abstracts it finds (saved by default at ./../output/all_DOIs.csv).

### 2. Create a Label Studio Project. 
Label Studio is the core tool used in this project for training NERs. It is free to use and can be downloaded and installed on any system. Once you have made a Label Studio account and installed it, you will again need the API key which will be entered into the config file under *ls_api_key*. Before you run any of the following code to initialize, edit, or interface with your Label Studio project make sure it is running on your machine. 

The first step is to create a new project in Label Studio, using the GUI (i.e. in your web browser). Name it however you want, and then make sure you keep track of the project ID. Then in the config file, find the *project_id* line and enter it here. This will need to be change if you ever create or run another project. 

In order to define the labels you want to use, the file *label_config.xml* needs to be set up for your project. The default one for this project is already confiugred. Feel free to use it, or use it as a template by adding or removing entries. 

The first time that you run this module, run **ls_make_labels.py** which needs the LS_API_KEY, LABEL_STUDIO_URL, PROJECT_ID so be sure these are set. Running this for the first time will create the label structure in your Label Studio project. You should see this once you refresh the project. 

### 3. Add abstracts to Label Studio
Next, run **ls_new_abstracts.py** which will access the downloaded DOI file and retrieve all of the corresponding abstracts. It will then upload these abstracts to your Label Studio project. 

Now switch over again to the Label Studio project. Here, I suggest you check out a brief tutorial on using Label Studio. But essentially what you are doing is annotating the abstracts that now appear in your project. As you scroll through a text block, highlight words that you would like the model to recognize as one of your chosen labels. Note, the models only work with 1:1 correspondence, so don't try to label a word with multiple labels. 

### 4. Label, update NER, predict labels, repeat
My recommendation is to annotate about 50 of these the first time, if you are starting from scratch. Then you can train/update your first NER. The trained NER is extremely helpful for labelling because you can use it to generate predictions, which will appear in label studio. Then as you continue to label more abstracts you will see the predictions get better and better, and as they get better you will spend less of your own time correcting the labels. 

Run **update_NER.py** which will retrain your NER on the annotations. If this is the FIRST TIME you run this and you are not using the custom NERs here, or there is some other reason you would like to start from scratch, run update_NER.py with the *--new* flag (py -3.8 update_NER.py --new) to have the program start with a default, publicly available NER. 

Then, you can run **ls_predict_labels.py** to generate new predictions. This calls **host_NER_model.py**, which starts an instance of your NER model on the machine, then feeds the fresh abstracts to it for predictions. Once it has made predictions, **ls_predict_labels.py** automatically uploads these to your Label Studio project for your convenience. 

I recommend training the NER on at least 200 abstracts before I would really trust it. The models in *models/* have been trained on 800+, which is still a somewhat small number in the scheme of NERs. That means that the NER will still make mistakes, although its error rate is fairly low already at this threshold (<10%). 

### 5. Scrape Abstracts
The final step is to generate data with **scrape_abstracts.py,** which will run through the abstracts, label words, categorize them and then output all of the results in a big table in /output/abstract_parsing1.csv  

### 6. Analyze Abstracts
The table of words taken from the abstracts is already useful for helping us sort and categorize papers in a more refined way. See the code in *analysis_in_R.* In particular at this stage, **abstract_table_summary.R** contains example R code that will plot papers according to certain key words. At the moment, most of the code here is used to distinguish papers that are more ecological in nature from those that are focused on agricultural or soil remediation practices. 

If you would like to use the results of analyzing the abstracts to create a subset of papers which are analyzed in subsequent steps, then

