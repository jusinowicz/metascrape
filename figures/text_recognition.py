#####Need to edit this around img_text. This is the non AWS, OCR version
#####Then need to make all of the other code follow from this output only.
#####
#####There are two different versions of the getProbableLabels functions. 
#####Ultimately for data extraction need the one in data_extraction/DataExtraction.
#####Make sure to do this for all of the functions! Now that I know about htis 
#####sloppy function naming. 

import cv2, json
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

# Directory of images to run the code on
img_dir = './download/images'
 
# Directory to save the output images
save_dir = './../output/figures'

#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed


def expand(points, margin = 1):
	return np.array([
		[[points[0][0][0] - margin, points[0][0][1] - margin]],
		[[points[1][0][0] + margin, points[1][0][1] - margin]],
		[[points[2][0][0] + margin, points[2][0][1] + margin]],
		[[points[3][0][0] - margin, points[3][0][1] + margin]]])


def detectText(path, image, image_text, img_text):
	# Get image dimensions
	img_height, img_width, _ = image.shape
	
	# Convert image to grayscale (optional, improves accuracy)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect text using Tesseract
	custom_config = r'--oem 3 --psm 6'  # Default OCR settings, you can tweak as needed
	data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
	
	if path.name not in image_text:
		image_text[path.name] = {}
		image_text[path.name]['TextDetections'] = []
		
	if path.name not in img_text:
		img_text[path.name] = []
		
	# Iterate over detected text blocks
	n_boxes = len(data['text'])
	for i in range(n_boxes):
		if int(data['conf'][i]) >= 80:  # Filter low-confidence detections
			detected_text = data['text'][i]
			(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
			
			# Store the detected text and its bounding box
			image_text[path.name]['TextDetections'].append({
				'DetectedText': detected_text,
				'BoundingBox': (x, y, w, h),
				'Confidence': data['conf'][i]
			})
			
			# Draw a rectangle around the detected text
			vertices = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.int32)
			image = cv2.fillPoly(image, vertices, (255, 255, 255))
			
			# Append to img_text for further processing
			img_text[path.name].append(
				(
					detected_text,
					(x, y, w, h)
				)
			)
			
	return image

image_text = {}
img_text = {}

for subfolder in Path(img_dir).iterdir():
	if subfolder.is_dir():  # Check if it's a directory (subfolder)
		# Now iterate over files within this subfolder
		# for file in subfolder.iterdir():
		for index, path in enumerate(Path(subfolder).iterdir()):
			if path.suffix.lower() in image_extensions:  # Check if it's an image
				#filepath = str(file)  # Get the full path as a string
				filepath = path
				print("[{0}] file name: {1}".format(index, path.name))
		
				image = cv2.imread(filepath)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)	
				image = detectText(path, image, image_text, img_text)
				detectText(path, image, img_text)
	
with open('./json/pytess-image-text.json', 'w') as out:
    json.dump(image_text, out)

with open('./json/ocr-image-text.json', 'w') as out:
	json.dump(img_text, out)