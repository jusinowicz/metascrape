"""
Update the NER with user-annotated data from Label Studio
- Check the configuration file config_fulltext.csv for configuration.
- Config needs: 
		model_load_dir: The location of the custom NER
		model_save_dir: Where to save the fitted/updated NER
		latest_labels: Directory of Label Studio projects. 
			Default is ./../label_studio_projects/ 
- This module does not interface directly with Label Studio. HOWEVER
	it does require the latest annotations from a project as a .json
	file, stored in the label_studio_projects directory. This can be 
	easily exported from within Label Studio. 
"""
#libraries
try:
	import os
	import csv
	import sys
	import json
	import argparse

	#Spacy NER
	import spacy
	from spacy.training.example import Example
	
	#the custom modules
	sys.path.append(os.path.abspath('./../'))
	from common.config import load_config, get_config_param, ConfigError
	from common.utilities import clean_annotations
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
		model_load_dir = get_config_param(config, 'model_load_dir', required=True)
		model_save_dir = get_config_param(config, 'model_save_dir', required=True)
		latest_labels = get_config_param(config, 'latest_labels', required=True)
		print("Config_abstracts successfully loaded")
	except ConfigError as e:
		print(f"Configuration error: {e}")
	except Exception as e:
		print(f"An error occurred: {e}")

	try:
		#Get the list of project files. 
		current_json = {f for f in os.listdir(latest_labels) if f.endswith('.json')}
		current_json = list(current_json)
		#Order by creation to get the newest annotations
		current_json.sort(key=lambda f: os.path.getmtime(os.path.join(latest_labels, f)), reverse=True)
		ll = latest_labels+current_json[0]

		# Load the exported data from Label Studio
		with open(ll, 'r', encoding='utf-8') as file:
			labeled_data = json.load(file)

		# Filter out entries that have not been annotated by a human: 
		labeled_data = [task for task in labeled_data if 'annotations' in task and task['annotations']]

		# Clean Annotations Function
		cleaned_data = clean_annotations(labeled_data)

		if args.new:
			#Train fresh. Load a default spaCy NLP
			nlp = spacy.load("en_core_sci_md")
			print("Train fresh. Load a default spaCy NLP")
		else:
			#Retrain an existing custom NER
			nlp = spacy.load(model_load_dir)
			print(f"Retraining {model_load_dir}")

		# Prepare the data for spaCy
		examples = []
		for text, annotations in cleaned_data:
		    doc = nlp.make_doc(text)
		    example = Example.from_dict(doc, annotations)
		    examples.append(example)

		# Add the new labels to the NER component
		ner = nlp.get_pipe("ner")
		labels = set(label for _, anns in cleaned_data for _, _, label in anns["entities"])
		for label in labels:
		    ner.add_label(label)

		# Disable other pipes for training
		print("Annotations parsed. Begin training: ")
		unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
		with nlp.disable_pipes(*unaffected_pipes):
		    optimizer = nlp.resume_training()
		    for i in range(50):  # Number of iterations
		        print(f"Iteration {i+1}")
		        losses = {}
		        nlp.update(
		            examples,
		            drop=0.35,  # Dropout - make it harder to memorize data
		            losses=losses,
		        )
		        print(losses)

		# Save the model
		nlp.to_disk(model_save_dir)
		print(f"Model saved to {model_save_dir}")

	except Exception as e:
		print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()