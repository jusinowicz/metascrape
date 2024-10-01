import cv2, json, os, sys, re, statistics
import pytesseract, easyocr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#the custom modules
sys.path.append(os.path.abspath('./'))
from text_utils import getYVal, getProbableLabels, addToExcel, findMaxConsecutiveOnes, detectAxes
from text_utils import getRatio 
from text_utils import filterBbox, boxGroup, mergeRects, RectDist, lineIntersectsRectX


#Definitions of images
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.pdf']  # Add more as needed

#Default image scale
scl = 5

with open('./aws-rekognition-output.json') as awshandler:
	bb1 = json.load(awshandler)

def expand(points, margin = 1):
	return np.array([
		[[points[0][0][0] - margin, points[0][0][1] - margin]],
		[[points[1][0][0] + margin, points[1][0][1] - margin]],
		[[points[2][0][0] + margin, points[2][0][1] + margin]],
		[[points[3][0][0] - margin, points[3][0][1] + margin]]])


def detectText_easyocr(path, image, image_text, img_text, scl = 3):
	# Load the image using OpenCV
	img = cv2.imread(path)
	height, width, _ = img.shape
	
	#Try some things to enhance image readibility 
	img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
	
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
image = detectText_easyocr(filepath, image, image_text, img_text, scl=scl)

texts = []
yValueDict = {}

images_text = img_text.copy() 
bbox_text = image_text.copy()
image_text = images_text[filepath.name]
texts = bbox_text[filepath.name]['TextDetections']
img = cv2.imread(filepath)                                                                       
img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC) 
xaxis, yaxis = detectAxes(filepath, scl=scl)
index = 0

##############
#yValueDict = getYVal(index, filepath, yValueDict, image_text, texts, image_extensions)
##############

###############################################################################
#This is the getYVal code
img = cv2.imread(filepath)
img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_height, img_width, _ = img.shape

# Axes detection
xaxis, yaxis = detectAxes(filepath, scl=scl)

for (x1, y1, x2, y2) in [xaxis]:
	xaxis = (x1, y1, x2, y2)

for (x1, y1, x2, y2) in [yaxis]:
	yaxis = (x1, y1, x2, y2)

img, x_labels, x_labels_list, _, _, _, _, legends, legendBoxes = getProbableLabels(img, image_text, xaxis, yaxis)
actual_image = img.copy()


list_text, normalize_ratio = getRatio(filepath, image_text, xaxis, yaxis, scl=scl)
print("[getYVal] legends: {0}".format(legends))
print("[{0}] path: {1}, ratio: {2}".format(index, filepath.name, normalize_ratio), end='\n\n')

for text in texts:
	if text['Type'] == 'WORD' and text['Confidence'] >= 0:
		vertices = [[vertex['X'] * img_width, vertex['Y'] * img_height] for vertex in text['Geometry']['Polygon']]
		vertices = np.array(vertices, np.int32)
		vertices = vertices.reshape((-1, 1, 2))
		img = cv2.fillPoly(img, [expand(vertices, 1)], (255, 255, 255))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if cv2.contourArea(contour) < 0.01 * img_height * img_width]
contours = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
rects = [cv2.boundingRect(contour) for contour in contours]
groups = []
legendtexts = []
legendrects = []

for box in legendBoxes:
	text, (textx, texty, width, height) = box
	bboxes = filterBbox(rects, box)
	
	if bboxes is not None:
		for rect in [bboxes]:
			(x, y, w, h) = rect
			legendrects.append(rect)
			
			group = boxGroup(actual_image, rect)[0]
			group = [arr.tolist() for arr in group]
			
			groups.append(group)
			legendtexts.append(text)
			
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
		cv2.rectangle(img, (textx, texty), (textx + width, texty + height), (255, 0, 0), 2)
				 

data = {}
for legend in legends:
	data[legend] = {}
	
	for x_label, box in x_labels_list:
		data[legend][x_label] = 0.0

##		
for i in range(len(groups)):
	##
	img = cv2.imread(filepath)
	img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	legendtext = legendtexts[i]
	
	for box in legendrects:
		(textx, texty, width, height) = box
		cv2.rectangle(img, (textx, texty), (textx + width, texty + height), (255, 255, 255), cv2.FILLED)
		
	mask = None
	for value in groups[i]:
		COLOR_MIN = np.array([value[0], value[1], value[2]], np.uint8)
		COLOR_MAX = np.array([value[0], value[1], value[2]], np.uint8)
		if mask is None:
			mask = cv2.inRange(img, COLOR_MIN, COLOR_MAX)
		else:
			mask = mask | cv2.inRange(img, COLOR_MIN, COLOR_MAX)
			
	image = cv2.bitwise_and(img, img, mask = mask)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (3, 3))
	edged = cv2.Canny(image, 0, 250)
	contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = [contour for contour in contours if cv2.contourArea(contour) > 0.]
	# Remove noisy ones!
	# if len(contours) == 0 or len(contours) > 100:
	# 	continue
	
	contours = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
	print(f"This is rects 1 {len(rects)}")
	rects = mergeRects(contours)
	print(f"This is rects 2 {len(rects)}")
	#height_threshold = 0.5 * max([rect[3] for rect in rects])  # 20% of the max height
	height_threshold = statistics.median([rect[3] for rect in rects])
	rects = [rect for rect in rects if rect[3] >= height_threshold]	
	print(f"This is rects 3{len(rects)}")
	
	textBoxes = []
	labels = []
	for rectBox in rects:
		min_distance = sys.maxsize
		closestBox = None
		labeltext = None
		
		for text, textBox in x_labels_list:
			if RectDist(rectBox, textBox) < min_distance:
				closestBox = textBox
				min_distance = RectDist(rectBox, textBox)
				labeltext = text
		textBoxes.append(closestBox)
		labels.append(labeltext)
		
	list_len = []
	
	for rect in rects:
		list_len.append((rect, float(rect[3])))
		
	# y-values will be a product of the normalize ratio and each length              
	y_val = [(rect, round(l* normalize_ratio, 4)) for rect, l in list_len]
	
	# Create a figure and axis to plot
	fig, ax = plt.subplots(1)
	
	# Display the image
	ax.imshow(img)
	
	for x_label, box in x_labels_list:
		(x, y, w, h) = box
		value = 0.0
		closest = None
		dist = sys.maxsize
		# Create a rectangle patch for the bounding box
		rect1 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
		# Add the rectangle patch to the plot
		ax.add_patch(rect1)
		# Optionally, add the label for each box
		ax.text(x, y - 5, x_label, color='r', fontsize=12, backgroundcolor='w')
		
		for index, item in enumerate(y_val):
			if labels[index] == x_label:
				(vx, vy, vw, vh) = item[0]
				# Create a rectangle patch for the bounding box
				
				if abs(x + w/2 - vx - vw/2) < dist:
					# print(f"This is index {index} and item {item} with coords {(vx, vy, vw, vh)}")
					# print(f"This is the condition:{abs(x + w/2 - vx - vw/2)} and {dist}")
					dist = abs(x + w/2 - vx - vw/2)
					closest = item[0]
					value = item[1]
					
		rect2 = patches.Rectangle((vx, vy), vw, vh, linewidth=2, edgecolor='b', facecolor='none')
		# Add the rectangle patch to the plot
		ax.add_patch(rect2)
		# Optionally, add the label for each y_val box
		ax.text(vx, vy - 5, f"Value: {value}", color='b', fontsize=10, backgroundcolor='w')
		plt.draw()
		#plt.pause(3)
			
		#rect1.remove()
		#rect2.remove()			  
		data[legendtext][x_label] = value
	
	#yValueDict[filepath.name] = data


######################
#Visualize rectangles
###########################


def showRecs(filepath, legendtexts, groups, scl=scl):
	for i in range(len(groups)):
		## Read and preprocess the image
		img = cv2.imread(filepath)
		img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		
		# Adaptive thresholding or Otsu’s method for better contour detection
		_, threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		
		# Apply Morphological Transformations (Closing to fill gaps)
		kernel = np.ones((5, 5), np.uint8)
		threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
		
		# Optionally apply Canny edge detection for better contour extraction
		edges = cv2.Canny(threshold, 100, 200)
		
		# Find contours
		contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		
		# Filter contours based on area and aspect ratio
		img_height, img_width = img.shape[:2]
		filtered_rects = []
		for cnt in contours:
			x, y, w, h = cv2.boundingRect(cnt)
			filtered_rects.append((x, y, w, h))
			# area = cv2.contourArea(cnt)
			# if area < 0.01 * img_height * img_width:  # Adjust as needed
			# 	x, y, w, h = cv2.boundingRect(cnt)
			# 	aspect_ratio = float(w) / h
			# 	if 0.1 < aspect_ratio < 0.5:  # Bars should be more vertical than white space
			# 		filtered_rects.append((x, y, w, h))
					
		# Post-process contours (e.g., mergeRects function)
		rects = mergeRects(filtered_rects, mode='rects')
		
		# Sort rectangles by height (h), descending order
		rects.sort(key=lambda rect: rect[3], reverse=True)
		
		# (Optional) Visualize bounding boxes for debugging purposes
		for rect in rects:
			x, y, w, h = rect
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			
		# Save or display the image with bounding boxes
		plt.imshow(img)
		plt.axis('off')  # Turn off axis numbers and ticks
		plt.title(f"Bounding Boxes for Group {i+1}")
		plt.show()
		
		## Legend text and further processing (depending on your groups logic)
		legendtext = legendtexts[i]
		# Add more processing logic for y-values based on rects
		
		print(f"This is rects {i+1}: {len(rects)}")
	
	return rects
	

a1 = showRecs(filepath, legendtexts, groups, scl=scl)








#######################################################
#Single loop version
########################################################
for i in range(len(groups)):
	##
img = cv2.imread(filepath)
img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
legendtext = legendtexts[i]

for box in legendrects:
	(textx, texty, width, height) = box
	cv2.rectangle(img, (textx, texty), (textx + width, texty + height), (255, 255, 255), cv2.FILLED)

mask = None
for value in groups[i]:
	COLOR_MIN = np.array([value[0], value[1], value[2]], np.uint8)
	COLOR_MAX = np.array([value[0], value[1], value[2]], np.uint8)
	if mask is None:
		mask = cv2.inRange(img, COLOR_MIN, COLOR_MAX)
	else:
		mask = mask | cv2.inRange(img, COLOR_MIN, COLOR_MAX)
		

image = cv2.bitwise_and(img, img, mask = mask)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, (3, 3))
edged = cv2.Canny(image, 0, 250)
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [contour for contour in contours if cv2.contourArea(contour) > 0.]
# Remove noisy ones!
# if len(contours) == 0 or len(contours) > 100:
# 	continue

contours = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
rects = mergeRects(contours)
#height_threshold = 0.5 * max([rect[3] for rect in rects])  # 20% of the max height
height_threshold = 0.5 * statistics.median([rect[3] for rect in rects])
rects = [rect for rect in rects if rect[3] >= height_threshold]	


textBoxes = []
labels = []
for rectBox in rects:
	min_distance = sys.maxsize
	closestBox = None
	labeltext = None
	for text, textBox in x_labels_list:
		if RectDist(rectBox, textBox) < min_distance:
			closestBox = textBox
			min_distance = RectDist(rectBox, textBox)
			labeltext = text
	textBoxes.append(closestBox)
	labels.append(labeltext)

list_len = []
	
for rect in rects:
	list_len.append((rect, float(rect[3])))

# y-values will be a product of the normalize ratio and each length              
y_val = [(rect, round(l* normalize_ratio, 4)) for rect, l in list_len]

# Create a figure and axis to plot
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(img)

for x_label, box in x_labels_list:
	(x, y, w, h) = box
	value = 0.0
	closest = None
	dist = sys.maxsize
	# Create a rectangle patch for the bounding box
	rect1 = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
	# Add the rectangle patch to the plot
	ax.add_patch(rect1)
	# Optionally, add the label for each box
	ax.text(x, y - 5, x_label, color='r', fontsize=12, backgroundcolor='w')
	
	for index, item in enumerate(y_val):
		if labels[index] == x_label:
			(vx, vy, vw, vh) = item[0]
			# Create a rectangle patch for the bounding box
			
			if abs(x + w/2 - vx - vw/2) < dist:
				print(f"This is index {index} and item {item} with coords {(vx, vy, vw, vh)}")
				print(f"This is the condition:{abs(x + w/2 - vx - vw/2)} and {dist}")
				dist = abs(x + w/2 - vx - vw/2)
				closest = item[0]
				value = item[1]
				
	rect2 = patches.Rectangle((vx, vy), vw, vh, linewidth=2, edgecolor='b', facecolor='none')
	# Add the rectangle patch to the plot
	ax.add_patch(rect2)
	# Optionally, add the label for each y_val box
	ax.text(vx, vy - 5, f"Value: {value}", color='b', fontsize=10, backgroundcolor='w')
	plt.draw()
	plt.pause(3)
		
	#rect1.remove()
	#rect2.remove()			  
	data[legendtext][x_label] = value
	
yValueDict[path.name] = data

# Iterate through the list_len, which contains tuples of (rect, l)
# for rect, l in list_len:
# 	# Print the original values for debugging
# 	print(f"Original rect: {rect}, Original l: {l}")
	
# 	# Calculate l * normalize_ratio
# 	scaled_value = l * normalize_ratio
# 	print(f"Scaled value (l * normalize_ratio): {scaled_value}")
	
# 	# Round the scaled value to 1 decimal place
# 	rounded_value = round(scaled_value, 1)
# 	print(f"Rounded value: {rounded_value}")
	
# 	# Create a tuple with the rect and the rounded value, and append it to y_val
# 	y_val.append((rect, rounded_value))





#####
# Create a figure and axis to plot
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(image_rgb)

# Plot bounding boxes from x_labels_list
for x_label, box in x_labels_list:
	(x, y, w, h) = box
	# Create a rectangle patch for the bounding box
	rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
	# Add the rectangle patch to the plot
	ax.add_patch(rect)
	# Optionally, add the label for each box
	ax.text(x, y - 5, x_label, color='r', fontsize=12, backgroundcolor='w')

# Plot bounding boxes from y_val (if they contain bounding boxes as well)
for index, item in enumerate(y_val):
	(vx, vy, vw, vh) = item[0]  # Assuming item[0] contains the bounding box
	# Create a rectangle patch for the bounding box
	rect = patches.Rectangle((vx, vy), vw, vh, linewidth=2, edgecolor='b', facecolor='none')
	# Add the rectangle patch to the plot
	ax.add_patch(rect)
	# Optionally, add the label for each y_val box
	ax.text(vx, vy - 5, f"Value: {item[1]}", color='b', fontsize=10, backgroundcolor='w')

# Show the final plot with bounding boxes
plt.show()

###############################################################################
###############################################################################
###############################################################################
# Notes not in progress

y_text = []

for (x1, y1, x2, y2) in [xaxis]:
	xaxis = (x1, y1, x2, y2)

for (x1, y1, x2, y2) in [yaxis]:
	yaxis = (x1, y1, x2, y2)

img, x_labels, _, x_text, y_labels, y_labels_list, y_text_list, legends, _ = getProbableLabels(img,
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

data = yValueDict[filepath.name]

# Print the output here!
print("file name    :  ", filepath.name)
print("x-text       :  ", x_text)
print("x-labels     :  ", x_labels)
print("y-text       :  ", y_text)
print("y-labels     :  ", y_labels)
print("legends      :  ", legends)
print("data         :  ", data, end= "\n\n")



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
		

###############################################################################
#This section is the getRatio code.
###############################################################################
list_text = []
list_ticks = []

image = cv2.imread(filepath)
image = cv2.resize(image, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
	
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channels = image.shape

for text, (textx, texty, w, h) in image_text:
	text = text.strip()
				
	(x1, y1, x2, y2) = xaxis
	(x11, y11, x22, y22) = yaxis
	
	# To the left of y-axis and top of x-axis
	if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
		np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == 1):
		
		# Consider numeric only for ticks on y-axis
		numbers = re.findall(r'\d+(?:\.\d+)?', text)
		if bool(numbers):
			list_text.append((numbers[0], (textx, texty, w, h)))
			

# Get the y-labels by finding the maximum
# intersections with the sweeping line
maxIntersection = 0
maxList = []
for i in range(x11):
	count = 0
	current = []
	for index, (text, rect) in enumerate(list_text):
		if lineIntersectsRectX(i, rect):
			count += 1
			current.append(list_text[index])
						
	if count > maxIntersection:
		maxIntersection = count
		maxList = current

# Get list of text and ticks
list_text = []
for text, (textx, texty, w, h) in maxList:
	list_text.append(float(text))
	list_ticks.append(float(texty + h))
	
text_sorted = (sorted(list_text))
ticks_sorted  = (sorted(list_ticks))

ticks_diff = ([ticks_sorted[i] - ticks_sorted[i-1] for i in range(1, len(ticks_sorted))])
text_diff = ([text_sorted[i] - text_sorted[i-1] for i in range(1, len(text_sorted))])
print("[get text-to-tick ratio] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))

# Detected text may not be perfect! Remove the outliers.
ticks_diff = reject_outliers(np.array(ticks_diff))
text_diff = reject_outliers(np.array(text_diff))
print("[reject_outliers] ticks_diff: {0}, text_diff: {1}".format(ticks_diff, text_diff))

normalize_ratio = np.array(text_diff).mean() / np.array(ticks_diff).mean()


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot_bounding_boxes(img_path, x_labels_list, rects):
	# Load and resize the image
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Create a figure and axis to plot
	fig, ax = plt.subplots(1, figsize=(10, 10))
	
	# Display the image
	ax.imshow(img)
	
	# Plot x_labels bounding boxes
	for x_label, box in x_labels_list:
		(x, y, w, h) = box
		rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
		ax.text(x, y - 5, x_label, color='r', fontsize=12, backgroundcolor='w')  # Add x-axis labels in red
	
	# Plot bounding rectangles for detected bars
	for (x, y, w, h) in rects:
		rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
		ax.add_patch(rect)
	
	# Optionally, you can label the detected bounding boxes
	for i, (x, y, w, h) in enumerate(rects):
		ax.text(x + w / 2, y - 5, f'Bar {i}', color='b', fontsize=10, backgroundcolor='w')  # Add bar labels in blue
	
	# Show the plot
	plt.show()

# Example usage
plot_bounding_boxes(filepath, x_labels_list, rects)
