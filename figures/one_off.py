import cv2, json, os, sys, re
import pytesseract, easyocr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams

#the custom modules
sys.path.append(os.path.abspath('./'))
from text_utils import getYVal, getProbableLabels, addToExcel, findMaxConsecutiveOnes, detectAxes


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


def detectText_easyocr(path, image, image_text, img_text):
	# Load the image using OpenCV
	img = cv2.imread(path)
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
	
	if path.name not in image_text:
		image_text[path.name] = {}
		image_text[path.name]['TextDetections'] = []
		
	if path.name not in img_text:
		img_text[path.name] = []
	
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
				'Type': 'WORD',  # Assuming line level detection, adjust if needed
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
			# Store the detected text and its bounding box
			image_text[path.name]['TextDetections'].append(detected_text_data)
			
			# Draw a rectangle around the detected text (optional for visualization)
			vertices = np.array([[[left, top], [left + width, top], [left + width, top + height], [left, top + height]]], np.int32)
			img = cv2.fillPoly(img, vertices, (255, 255, 255))
			
			# Append to img_text for further processing
			img_text[path.name].append(
				(
					text,
					(left, top, width, height)
				)
			)
			
	return image


# Directory of images to run the code on
img_dir = './download/images'
 
# Directory to save the output images
save_dir = './../output/figures'

#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed

a1 = Path(img_dir).iterdir() 
file = next(a1)
filepath = file

#This file should match this entry: 
list(bb1)[2]
for_comparison = bb1[list(bb1)[2]]
print(filepath)

#results = detect_text_opencv(filepath)

image_text = {}
img_text = {}     

image = cv2.imread(filepath)
image = detectText_easyocr(filepath, image, image_text, img_text)

texts = []
yValueDict = {}

images_text = img_text.copy() 
bbox_text = image_text.copy()
image_text = images_text[filepath.name]
texts = bbox_text[filepath.name]['TextDetections']
img = cv2.imread(filepath)                                                                       
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC) 
xaxis, yaxis = detectAxes(filepath)
yValueDict = getYVal(index, filepath, yValueDict, image_text, texts, image_extensions)


img, x_labels, x_labels_list, _, _, _, _, legends, legendBoxes = getProbableLabels(img, image_text, xaxis, yaxis)
actual_image = img.copy()

for text, (textx, texty, w, h) in image_text:
    text = text.strip()
    print(f"Text is {text}, textx is {textx}, texty is {texty}, and w h is {w,h}")
    print(f"First number is {np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1))}")
    print(f"Second number is {np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) }")


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
	if text['Type'] == 'WORD' and text['Confidence'] >= 0.5:
		print(f"This is the word {text['DetectedText']}")
		vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
		vertices = np.array(vertices, np.int32)
		vertices = vertices.reshape((-1, 1, 2))
		img = cv2.fillPoly(img, [expand(vertices, 1)], (255, 255, 255))
		

#def getProbableLabels(image, image_text, xaxis, yaxis):
y_labels = []
x_labels = []
legends = []
y_text_list = []

height, width, channels = image.shape

(x1, y1, x2, y2) = xaxis
(x11, y11, x22, y22) = yaxis

image_text = cleanText(image_text)

for text, (textx, texty, w, h) in image_text:
	text = text.strip()
	print(f"Text is {text}, textx is {textx}, texty is {texty}, and w h is {w,h}")
	print(f"First number is {np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1))}")
	print(f"Second number is {np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11))}")
	
	# To the left of y-axis and top of x-axis
	if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
		np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == 1):
		
		numbers = re.findall(r'^[+-]?\d+(?:\.\d+)?[%-]?$', text)
		if bool(numbers):
			y_labels.append((text, (textx, texty, w, h)))
		else:
			y_text_list.append((text, (textx, texty, w, h)))
		
	# To the right of y-axis and bottom of x-axis
	elif (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == 1 and
		np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
		x_labels.append((text, (textx, texty, w, h)))
		
	# Top of x-axis and to the right of y-axis
	elif (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
		np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
		
		# Consider non-numeric only for legends
		legends.append((text, (textx, texty, w, h)))


def canMerge(group, candidate):
	candText, candRect = candidate
	candx, candy, candw, candh = candRect
	
	for memText, memRect in group:
		memx, memy, memw, memh = memRect
			
		if abs(candy - memy) <= 5 and abs(candy + candh - memy - memh) <= 5:
			return True
		elif abs(candx - memx) <= 5:
			return True
			
	return False

# Grouping Algorithm
legend_groups = []
for index, (text, rect) in enumerate(legends):
	#print("text: {0}, rect: {1}\n".format(text, rect))
	
	for groupid, group in enumerate(legend_groups):
		if canMerge(group, (text, rect)):
			group.append((text, rect))
			break
	else:
		legend_groups.append([(text, rect)])

#print(legend_groups)
#print("\n\n")

maxList = []

if len(legend_groups) > 0:
	maxList = max(legend_groups, key = len)
	
legends = []
for text, (textx, texty, w, h) in maxList:
	legends.append(text)
	
return image, x_labels, x_labels_list, x_text, y_labels, y_labels_list, y_text_list, legends, maxList
