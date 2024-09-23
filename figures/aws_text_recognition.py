#####Need to edit this around img_text. This is the non AWS, OCR version
#####Then need to make all of the other code follow from this output only.
##### SEE text_recognition FOR THE pytesseract VERSION OF THIS FUNCTION!!!

#####There are two different versions of the getProbableLabels functions. 
#####Ultimately for data extraction need the one in data_extraction/DataExtraction.
#####Make sure to do this for all of the functions! Now that I know about htis 
#####sloppy function naming. 

import cv2, json, boto3
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


#Detect Text function 
def detectText(path, image, image_text, img_text):
	
	img_height, img_width, channels = image.shape
	_, im_buf = cv2.imencode("." + path.name.split(".")[-1], image)
		
	response = client.detect_text(
		Image = {
			"Bytes" : im_buf.tobytes()
		}
	)
	
	if path.name not in image_text:
		image_text[path.name] = {}
		image_text[path.name]['TextDetections'] = response['TextDetections']
	else:
		image_text[path.name]['TextDetections'].extend(response['TextDetections'])
		
	textDetections = response['TextDetections']
		
	if path.name not in img_text:
		img_text[path.name] = []
			
	for text in textDetections:
		if text['Type'] == 'WORD' and text['Confidence'] >= 80:
				
			vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
			vertices = np.array(vertices, np.int32)
			vertices = vertices.reshape((-1, 1, 2))
			
			image = cv2.fillPoly(image, [expand(vertices)], (255, 255, 255))
				  
			left = np.amin(vertices, axis=0)[0][0]
			top = np.amin(vertices, axis=0)[0][1]
			right = np.amax(vertices, axis=0)[0][0]
			bottom = np.amax(vertices, axis=0)[0][1]
			
			img_text[path.name].append(
				(
					text['DetectedText'],
					(
						int(left),
						int(top),
						int(right - left),
						int(bottom - top)
					)
				)
			)
	return image

img_text = {}
image_text = {}
client = boto3.client('rekognition', region_name='us-west-2')

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
				detectText(path, image, image_text, img_text)



with open('../data/aws-rekognition-output.json', 'w') as out:
	json.dump(image_text, out)
	
with open('../data/ocr-image-text.json', 'w') as out:
	json.dump(img_text, out)