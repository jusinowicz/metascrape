import cv2, imutils, re, sys, math
import xlsxwriter, json, os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

#the custom modules
sys.path.append(os.path.abspath('./'))
from axes_utils import findMaxConsecutiveOnes, detectAxes
from text_utils import getYVal, getProbableLabels, addToExcel

# Directory of images to run the code on
img_dir = './download/images'

# Directory to save the output images
save_dir = './../output/figures'


#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed

with open('./aws-rekognition-output.json') as awshandler:
    bb1 = json.load(awshandler)

images = []
texts = []
yValueDict = {}

#Write to workbook
workbook = xlsxwriter.Workbook('../results/FigureData1.xlsx', {'nan_inf_to_errors': True})

for subfolder in Path(img_dir).iterdir():
	if subfolder.is_dir():  # Check if it's a directory (subfolder)
		# Now iterate over files within this subfolder
		# for file in subfolder.iterdir():
		for index, path in enumerate(Path(subfolder).iterdir()):
			if path.suffix.lower() in image_extensions:  # Check if it's an image
				#filepath = str(file)  # Get the full path as a string
				filepath = path
				print("[{0}] path: {1}".format(index, path.name))
				image_text = images_text[path.name]
				texts = bbox_text[path.name]['TextDetections']
				
				yValueDict = getYVal(index, filepath, yValueDict, image_text, texts, image_extensions)
				
				image = cv2.imread(filepath)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				
				height, width, channels = image.shape
				xaxis, yaxis = detectAxes(filepath)
				y_text = []

				for (x1, y1, x2, y2) in [xaxis]:
					xaxis = (x1, y1, x2, y2)

				for (x1, y1, x2, y2) in [yaxis]:
					yaxis = (x1, y1, x2, y2)
					
				image, x_labels, _, x_text, y_labels, y_labels_list, y_text_list, legends, _ = getProbableLabels(image,
																									  image_text,
																									  xaxis,
																									  yaxis)
							
				# Sort bounding rects by y coordinate
				def getYFromRect(item):
					return item[1][1]

				y_labels_list.sort(key = getYFromRect)
				y_text_list.sort(key = getYFromRect, reverse=True)
				
				for text, (textx, texty, w, h) in y_text_list:
					y_text.append(text)

				# Append doi values for each image.
				pdfname = '-'.join(path.name.split('-')[:-2]) + '.pdf'
				
				if pdfname in doidata['Doi']:
					doi = doidata['Doi'][pdfname]
				else:
					doi = ''
				 
				# Write to Excel
				worksheet = workbook.add_worksheet()            
				
				addToExcel("doi", doi, 0)
				addToExcel("file name", [path.name], 1)
				addToExcel("x-text", x_text, 2)
				addToExcel("x-labels", x_labels, 3)
				addToExcel("y-text", y_text, 4)
				addToExcel("y-labels", y_labels, 5)
				addToExcel("legends", legends, 6)
				
				data = yValueDict[path.name]

				column = 9
				for legend, datadict in data.items():
					if column == 9:
						addToExcel("", datadict.keys(), 8)    
						
					addToExcel(legend, datadict.values(), column)
					column += 1
				
				# Print the output here!
				print("file name    :  ", path.name)
				print("doi          :  ", doi)
				print("x-text       :  ", x_text)
				print("x-labels     :  ", x_labels)
				print("y-text       :  ", y_text)
				print("y-labels     :  ", y_labels)
				print("legends      :  ", legends)
				print("data         :  ", data, end= "\n\n")
				
				# Insert the image
				worksheet.insert_image('J21', filepath)

# Close the excel workbook!
workbook.close()