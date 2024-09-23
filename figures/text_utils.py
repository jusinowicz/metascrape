import cv2, imutils, re, xlsxwriter, json
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from pathlib import Path
from matplotlib import rcParams
from pytesseract import Output

# Directory of images to run the code on
img_dir = './download/images'

# Directory to save the output images
save_dir = './../output/figures'

def getTextFromImage(filepath, bw=False, debug=False):
	image_text = []
	
	image = cv2.imread(filepath)
	height, width, _ = image.shape
		
	if bw:
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# define range of black color in HSV
		lower_val = np.array([0, 0, 0])
		upper_val = np.array([179, 255, 179])

		# Threshold the HSV image to get only black colors
		mask = cv2.inRange(hsv, lower_val, upper_val)

		# Bitwise-AND mask and original image
		res = cv2.bitwise_and(image, image, mask = mask)

		# invert the mask to get black letters on white background
		image = cv2.bitwise_not(mask)
			
	d = pytesseract.image_to_data(image, config = "-l eng --oem 1 --psm 11", output_type = Output.DICT)
	n_boxes = len(d['text'])

	# Pick only the positive confidence boxes
	for i in range(n_boxes):
			
		if int(d['conf'][i]) >= 0:
				
			text = d['text'][i].strip()
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			image_text.append((d['text'][i], (x, y, w, h)))
	 
	if bw:  
		image = cv2.imread(filepath)
		image_text = list(set(image_text))
		white_bg = 255 * np.ones_like(image)
		
		for text, (textx, texty, w, h) in image_text:
			roi = image[texty:texty + h, textx:textx + w]
			white_bg[texty:texty + h, textx:textx + w] = roi
			
		image_text = []
		d = pytesseract.image_to_data(white_bg, config = "-l eng --oem 1 --psm 11", output_type = Output.DICT)
		n_boxes = len(d['text'])

		# Pick only the positive confidence boxes
		for i in range(n_boxes):

			if int(d['conf'][i]) >= 0:

				text = d['text'][i].strip()
				(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
				image_text.append((d['text'][i], (x, y, w, h)))
		
	# Remove all the duplicates in (text, box) pairs
	return list(set(image_text))


def getProbableLabels(image, image_text, xaxis, yaxis):
	y_labels = []
	x_labels = []
	legends = []
	
	height, width, channels = image.shape
	
	for text, (textx, texty, w, h) in image_text:
		text = text.strip()
					
		(x1, y1, x2, y2) = xaxis
		(x11, y11, x22, y22) = yaxis
			
		# To the right of y-axis and bottom of x-axis
		if (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == 1 and
			np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
			x_labels.append((text, (textx, texty, w, h)))
			
		# Top of x-axis and to the right of y-axis
		elif (np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1)) == -1 and
			np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) == -1):
			
			# Consider non-numeric only for legends
			if not bool(re.findall(r'\b[\d\.\d]+\b', text)):
				legends.append((text, (textx, texty, w, h)))
				
	# Get the x-labels by finding the maximum
	# intersections with the sweeping line
	maxIntersection = 0
	maxList = []
	for i in range(y1, height):
		count = 0
		current = []
		for index, (text, rect) in enumerate(x_labels):
			if lineIntersectsRectY(i, rect):
				count += 1
				current.append(x_labels[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
	
	# Sort bounding rects by x coordinate
	def getYFromRect(item):
		return item[1]

	maxList.sort(key = getYFromRect)
	
	x_labels = []
	for text, (textx, texty, w, h) in maxList:
		x_labels.append(text)
		cv2.rectangle(image, (textx, texty), (textx + w, texty + h), (255, 0, 0), 2)
	
	# Get possible legend text
	# For this, we need to search both top to
	# bottom and also from left to right.
	maxIntersection = 0
	maxList = []
	for i in range(y1):
		count = 0
		current = []
		for index, (text, rect) in enumerate(legends):
			if lineIntersectsRectY(i, rect):
				count += 1
				current.append(legends[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
			
	for i in range(x11, width):
		count = 0
		current = []
		for index, (text, rect) in enumerate(legends):
			if lineIntersectsRectX(i, rect):
				count += 1
				current.append(legends[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
		
	legends = []
	legendBoxes = []
	for text, (textx, texty, w, h) in maxList:
		legends.append(text)
		legendBoxes.append((textx, texty, w, h))
		#cv2.rectangle(image, (textx, texty), (textx + w, texty + h), (255, 0, 255), 2)
	
	legendBoxes = mergeRects(legendBoxes)
	
	for (textx, texty, w, h) in legendBoxes:
		cv2.rectangle(image, (textx, texty), (textx + w, texty + h), (255, 0, 255), 2)
	
	print("number of clusters : ", len(legendBoxes))
		
	return image, x_labels, _, legends



def lineIntersectsRectX(candx, rect):
	(x, y, w, h) = rect
	
	if x <= candx <= x + w:
		return True
	else:
		return False
	


def lineIntersectsRectY(candy, rect):
	(x, y, w, h) = rect
	
	if y <= candy <= y + h:
		return True
	else:
		return False


def getTextFromImageArray(image, mode):
	image_text = []
	
	if mode == 'y-text':
		image = cv2.transpose(image)
		image = cv2.flip(image, flipCode = 1)
		config = "-l eng --oem 1 --psm 11"
	elif mode == 'y-labels':
		config = "-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=.0123456789"
	
	d = pytesseract.image_to_data(image, config = config, output_type = Output.DICT)
	
	n_boxes = len(d['text'])

	# Pick only the positive confidence boxes
	for i in range(n_boxes):
			
		if int(d['conf'][i]) >= 0:
				
			text = d['text'][i].strip()
			
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			image_text.append((d['text'][i], (x, y, w, h)))
			
	# Remove all the duplicates in (text, box) pairs
	return list(set(image_text))



def maskImageForwardPass(filepath, start_idx):
	if path.name.endswith('.png') or path.name.endswith('.jpg') or path.name.endswith('.jpeg'):

		filepath = img_dir + "/" + path.name
		image = cv2.imread(filepath)
		height, width, channels = image.shape
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		start_idx = 1
		while start_idx <= width:
			if sum(gray[:, start_idx] < 200) != 0:
				break
			else:
				start_idx += 1
				
		end_idx = start_idx
		while end_idx <= width:
			if sum(gray[:, end_idx] < 200) == 0:
				break
			else:
				end_idx += 1
				
		gray[:, 1:start_idx] = 255
		gray[:, end_idx:width] = 255
		
		return gray, start_idx, end_idx



def maskImageBackwardPass(filepath, end_idx):
	# if path.name.endswith('.png') or path.name.endswith('.jpg') or path.name.endswith('.jpeg'):
	# 	filepath = img_dir + "/" + path.name
	image = cv2.imread(filepath)
	height, width, channels = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	while end_idx > 0:
		if sum(gray[:, end_idx] < 200) == 0:
			break
		else:
			end_idx -= 1
	
	gray[:, end_idx:width] = 255
	
	return gray



#Writing to Excel workbook
def addToExcel(dataname, data, row):
	col = 0

	worksheet.write(row, col, dataname)
	for content in data:
		col += 1
		worksheet.write(row, col, content)



def nearbyRectangle(current, candidate, threshold):
	(currx, curry, currw, currh) = current
	(candx, candy, candw, candh) = candidate
	
	currxmin = currx
	currymin = curry
	currxmax = currx + currw
	currymax = curry + currh
	
	candxmin = candx
	candymin = candy
	candxmax = candx + candw
	candymax = candy + candh
	
	# If candidate is on top, and is close
	if candymax <= currymin and candymax + threshold >= currymin:
		return True
	
	# If candidate is on bottom and is close
	if candymin >= currymax and currymax + threshold >= candymin:
		return True
	
	# If intersecting at the top, merge it
	if candymax >= currymin and candymin <= currymin:
		return True
	
	# If intersecting at the bottom, merge it
	if currymax >= candymin and currymin <= candymin:
		return True
	
	# If intersecting on the sides or is inside, merge it
	if (candymin >= currymin and
		candymin <= currymax and
		candymax >= currymin and
		candymax <= currymax):
		return True
	
	return False


# Matching the ratio for final data extraction
# Y-val data: - The height of each bounding box is recorded by the help of the merging 
#rectangles during Cluster count estimation method. - Eventually, we used the ratio to 
#calculate the y-values as:

def mergeRects(contours):
	rects = []
	rectsUsed = []

	# Just initialize bounding rects and set all bools to false
	for cnt in contours:
		rects.append(cnt)
		#rects.append(cv2.boundingRect(cnt))
		rectsUsed.append(False)

	# Sort bounding rects by x coordinate
	def getXFromRect(item):
		return item[0]

	rects.sort(key = getXFromRect)

	# Array of accepted rects
	acceptedRects = []

	# Merge threshold for x coordinate distance
	xThr = 5
	yThr = 5

	# Iterate all initial bounding rects
	for supIdx, supVal in enumerate(rects):
		if (rectsUsed[supIdx] == False):

			# Initialize current rect
			currxMin = supVal[0]
			currxMax = supVal[0] + supVal[2]
			curryMin = supVal[1]
			curryMax = supVal[1] + supVal[3]

			# This bounding rect is used
			rectsUsed[supIdx] = True

			# Iterate all initial bounding rects
			# starting from the next
			for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):

				# Initialize merge candidate
				candxMin = subVal[0]
				candxMax = subVal[0] + subVal[2]
				candyMin = subVal[1]
				candyMax = subVal[1] + subVal[3]

				# Check if x distance between current rect
				# and merge candidate is small enough
				if (candxMin <= currxMax + xThr):

					if not nearbyRectangle((candxMin, candyMin, candxMax - candxMin, candyMax - candyMin),
										   (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin), yThr):
						break

					# Reset coordinates of current rect
					currxMax = candxMax
					curryMin = min(curryMin, candyMin)
					curryMax = max(curryMax, candyMax)

					# Merge candidate (bounding rect) is used
					rectsUsed[subIdx] = True
				else:
					break

			# No more merge candidates possible, accept current rect
			acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

	#for rect in acceptedRects:
	#    img = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (121, 11, 189), 2)
	
	return acceptedRects



def getProbableYLabels(image, contours, xaxis, yaxis):
	y_labels = []
	
	height, width, channels = image.shape
	
	(x1, y1, x2, y2) = xaxis
	(x11, y11, x22, y22) = yaxis
	
	# Get the y-labels by finding the maximum
	# intersections with the sweeping line
	maxIntersection = 0
	maxList = []
	for i in range(x11):
		count = 0
		current = []
		for index, rect in enumerate(contours):
			if lineIntersectsRectX(i, rect):
				count += 1
				current.append(contours[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
					
	return image, maxList