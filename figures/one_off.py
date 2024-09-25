import cv2, json
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed

with open('./aws-rekognition-output.json') as awshandler:
	bb1 = json.load(awshandler)

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



def detect_text_opencv(image_path):
	# Load the image using OpenCV
	img = cv2.imread(image_path)
	
	# Convert image to grayscale (optional, improves accuracy)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Apply Gaussian Blur to reduce noise
	#img = cv2.GaussianBlur(img, (5, 5), 0)
	#Adaptive Thresholding
	#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
	#                          cv2.THRESH_BINARY, 11, 2)
	
	# Detect text using Tesseract
	custom_config = r'--oem 3 --psm 11'  # Default OCR settings, you can tweak as needed
	details= pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
	
	# Run Tesseract to detect text along with detailed data
	# custom_config = r'--oem 3 --psm 6'  # OCR engine mode and page segmentation mode
	# details = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
	
	results = []
	
	for i in range(len(details['text'])):
		if int(details['conf'][i]) > 0:  # Only include results with positive confidence
			# Create a dictionary similar to the AWS output
			detected_text_data = {
				'DetectedText': details['text'][i],
				'Type': 'LINE',  # Assuming line level detection, adjust if needed
				'Id': i,
				'Confidence': float(details['conf'][i]),
				'Geometry': {
					'BoundingBox': {
						'Width': details['width'][i] / img.shape[1],  # Normalize width
						'Height': details['height'][i] / img.shape[0],  # Normalize height
						'Left': details['left'][i] / img.shape[1],  # Normalize left position
						'Top': details['top'][i] / img.shape[0]  # Normalize top position
					},
					'Polygon': [
						{'X': details['left'][i] / img.shape[1], 'Y': details['top'][i] / img.shape[0]},
						{'X': (details['left'][i] + details['width'][i]) / img.shape[1], 'Y': details['top'][i] / img.shape[0]},
						{'X': (details['left'][i] + details['width'][i]) / img.shape[1], 'Y': (details['top'][i] + details['height'][i]) / img.shape[0]},
						{'X': details['left'][i] / img.shape[1], 'Y': (details['top'][i] + details['height'][i]) / img.shape[0]}
					]
				}
			}
			results.append(detected_text_data)
	
	return results


def detect_text_opencv(image_path):
	# Load the image using OpenCV
	img = cv2.imread(image_path)
	height, width, _ = img.shape
	
	#Try some things to enhance image readibility 
	img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	
	# # Convert image to grayscale (optional, improves accuracy)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#_, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
	# define range of black color in HSV
	lower_val = np.array([0, 0, 0])
	upper_val = np.array([179, 255, 179])
		
	# Threshold the HSV image to get only black colors
	mask = cv2.inRange(hsv, lower_val, upper_val)
		
	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(img, img, mask = mask)
		
	# invert the mask to get black letters on white background
	img= cv2.bitwise_not(mask)
	
	reader = easyocr.Reader(['en'])
	horizontal_details = reader.readtext(img)
	
	# Rotate the image 90 degrees clockwise for vertical text detection
	rotated_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	
	# Detect vertical text
	vertical_details = reader.readtext(rotated_image)
	
	# Rotate bounding boxes back to original coordinates for vertical text
	for box in vertical_details:
		for point in box[0]:
			# Swap x and y coordinates and adjust for the rotated image size
			point[0], point[1] = img.shape[1] - point[1], point[0]
	
	details = horizontal_details #+ vertical_details
	results = []
	
	# Assume result is the output from reader.readtext(image)
	for i, (bbox, text, confidence) in enumerate(details):
		print(f"This is the text: {text} with confidence {float(confidence)}")
		if confidence > 0:  # Only include results with positive confidence
			# Get the bounding box (as 4 points: (x, y))
			x1, y1 = bbox[0]  # Top-left
			x2, y2 = bbox[1]  # Top-right
			x3, y3 = bbox[2]  # Bottom-right
			x4, y4 = bbox[3]  # Bottom-left
			
			# Calculate bounding box dimensions
			left = min(x1, x4)
			top = min(y1, y2)
			width = max(x2, x3) - left
			height = max(y3, y4) - top
			
			# Create a dictionary similar to the AWS output
			detected_text_data = {
				'DetectedText': text,
				'Type': 'LINE',  # Assuming line level detection, adjust if needed
				'Id': i,
				'Confidence': float(confidence),
				'Geometry': {
					'BoundingBox': {
						'Width': width / img.shape[1],  # Normalize width
						'Height': height / img.shape[0],  # Normalize height
						'Left': left / img.shape[1],  # Normalize left position
						'Top': top / img.shape[0]  # Normalize top position
					},
					'Polygon': [
						{'X': x1 / img.shape[1], 'Y': y1 / img.shape[0]},
						{'X': x2 / img.shape[1], 'Y': y2 / img.shape[0]},
						{'X': x3 / img.shape[1], 'Y': y3 / img.shape[0]},
						{'X': x4 / img.shape[1], 'Y': y4 / img.shape[0]}
					]
				}
			}
			results.append(detected_text_data)
	return results
		



def detectText2(path, image, image_text, img_text):
	# Get image dimensions
	img_height, img_width, _ = image.shape
	
	#Try some things to enhance image readibility 
	image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	
	# Convert image to grayscale (optional, improves accuracy)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect text using Tesseract
	custom_config = r'--oem 3 --psm 11'  # Default OCR settings, you can tweak as needed
	data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
	
	if path.name not in image_text:
		image_text[path.name] = {}
		image_text[path.name]['TextDetections'] = []
		
	if path.name not in img_text:
		img_text[path.name] = []
		
	# Iterate over detected text blocks
	results = []
	n_boxes = len(data['text'])
	for i in range(n_boxes):
		if int(data['conf'][i]) >= 0:  # Filter low-confidence detections
			
			detected_text = data['text'][i]
			(x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
			
			# Create a dictionary similar to the AWS output
			details = data
			print(f"This is the text: {detected_text} with confidence {float(details['conf'][i])}")
			
			print()
			detected_text_data = {
				'DetectedText': details['text'][i],
				'Type': 'LINE',  # Assuming line level detection, adjust if needed
				'Id': i,
				'Confidence': float(details['conf'][i]),
				'Geometry': {
					'BoundingBox': {
						'Width': details['width'][i] / image.shape[1],  # Normalize width
						'Height': details['height'][i] / image.shape[0],  # Normalize height
						'Left': details['left'][i] / image.shape[1],  # Normalize left position
						'Top': details['top'][i] / image.shape[0]  # Normalize top position
					},
					'Polygon': [
						{'X': details['left'][i] / image.shape[1], 'Y': details['top'][i] / image.shape[0]},
						{'X': (details['left'][i] + details['width'][i]) / image.shape[1], 'Y': details['top'][i] / image.shape[0]},
						{'X': (details['left'][i] + details['width'][i]) / image.shape[1], 'Y': (details['top'][i] + details['height'][i]) / image.shape[0]},
						{'X': details['left'][i] / image.shape[1], 'Y': (details['top'][i] + details['height'][i]) / image.shape[0]}
					]
				}
			}
			results.append(detected_text_data)
			# Store the detected text and its bounding box
			image_text[path.name]['TextDetections'].append(detected_text_data)
			
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
			
	return results


# Directory of images to run the code on
img_dir = './download/images'
 
# Directory to save the output images
save_dir = './../output/figures'

#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed

image_text = {}
img_text = {}     

a1 = Path(img_dir).iterdir() 
file = next(a1)
filepath = file

#This file should match this entry: 
list(bb1)[2]
for_comparison = bb1[list(bb1)[2]]
print(filepath)

results = detect_text_opencv(filepath)

image = cv2.imread(filepath)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_height, img_width, _ = image.shape
#image = detectText(filepath, image, image_text, img_text)
#image2 = detectText2(filepath, image, image_text, img_text)
img=image

#The right way
texts = bb1[filepath.name]['TextDetections']
for text in texts:
	if text['Type'] == 'WORD' and text['Confidence'] >= 80:
		print(f"This is the word {text['DetectedText']}")
		vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
		vertices = np.array(vertices, np.int32)
		vertices = vertices.reshape((-1, 1, 2))
		img = cv2.fillPoly(img, [expand(vertices, 1)], (255, 255, 255))

#The new way
texts = image_text[filepath.name]['TextDetections']
for text in texts:
	if text['Type'] == 'WORD' and text['Confidence'] >= 80:
		vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
		vertices = np.array(vertices, np.int32)
		vertices = vertices.reshape((-1, 1, 2))
		img = cv2.fillPoly(img, [expand(vertices, 1)], (255, 255, 255))