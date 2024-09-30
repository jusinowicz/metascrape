import cv2, imutils, re, xlsxwriter, json, sys
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from pathlib import Path
from matplotlib import rcParams
from pytesseract import Output

#Method to detect x and y axis
def findMaxConsecutiveOnes(nums) -> int:
	count = maxCount = 0
		
	for i in range(len(nums)):
		if nums[i] == 1:
			count += 1
		else:
			maxCount = max(count, maxCount)
			count = 0
				
	return max(count, maxCount)


def detectAxes(filepath, threshold=None, debug=False):
	if filepath is None:
		return None, None
		
	if threshold is None:
		threshold = 10
		
	image = cv2.imread(filepath)
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
	
	height, width, channels = image.shape
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Get the max-consecutive-ones for eah column in the bw image, and...
	# pick the "first" index that fall in [max - threshold, max + threshold]
	maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[:, idx] < 200) for idx in range(width)]
	start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
	while start_idx < width:
		if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
			maxindex = start_idx
			break
			
		start_idx += 1
		
	yaxis = (maxindex, 0, maxindex, height)
		
	if debug:
		fig, ax = plt.subplots(1, 2)
		
		ax[0].imshow(image)
		
		ax[1].plot(maxConsecutiveOnes, color = 'k')
		ax[1].axhline(y = max(maxConsecutiveOnes) - 10, color = 'r', linestyle = 'dashed')
		ax[1].axhline(y = max(maxConsecutiveOnes) + 10, color = 'r', linestyle = 'dashed')
		ax[1].vlines(x = maxindex, ymin = 0.0, ymax = maxConsecutiveOnes[maxindex], color = 'b', linewidth = 4)
		
		plt.show()
		
	# Get the max-consecutive-ones for eah row in the bw image, and...
	# pick the "last" index that fall in [max - threshold, max + threshold]
	maxConsecutiveOnes = [findMaxConsecutiveOnes(gray[idx, :] < 200) for idx in range(height)]
	start_idx, maxindex, maxcount = 0, 0, max(maxConsecutiveOnes)
	while start_idx < height:
		if abs(maxConsecutiveOnes[start_idx] - maxcount) <= threshold:
			maxindex = start_idx
		
		start_idx += 1
			
	cv2.line(image, (0, maxindex), (width, maxindex),  (255, 0, 0), 2)
	xaxis = (0, maxindex, width, maxindex)
	
	if debug:
		rcParams['figure.figsize'] = 15, 8
		
		fig, ax = plt.subplots(1, 1)
		ax.imshow(image, aspect = 'auto')
	return xaxis, yaxis



def cleanText(image_text):
	return [(text, (textx, texty, w, h)) for text, (textx, texty, w, h) in image_text if text.strip() != 'I']
	

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


def getTextFromImage(filepath, bw=False, debug=False):
	image_text = []
	
	image = cv2.imread(filepath)
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
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
		image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

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


def getProbableLabels(image, image_text, xaxis, yaxis):
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
		# print(f"Text is {text}, textx is {textx}, texty is {texty}, and w h is {w,h}")
		# print(f"First number is {np.sign((x2 - x1) * (texty - y1) - (y2 - y1) * (textx - x1))}")
		# print(f"Second number is {np.sign((x22 - x11) * (texty - y11) - (y22 - y11) * (textx - x11)) }")
		
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
	
	# Get the y-labels by finding the maximum
	# intersections with the sweeping line
	maxIntersection = 0
	maxList = []
	for i in range(x11):
		count = 0
		current = []
		for index, (text, rect) in enumerate(y_labels):
			if lineIntersectsRectX(i, rect):
				count += 1
				current.append(y_labels[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
	
	y_labels_list = maxList.copy()
	
	y_labels = []
	for text, (textx, texty, w, h) in maxList:
		y_labels.append(text)
		
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
			
	x_labels_list = maxList.copy()
	
	x_text = x_labels.copy()
	x_labels = []
	hmax = 0
	
	for text, (textx, texty, w, h) in maxList:
		x_labels.append(text)
		if texty + h > hmax:
			hmax = texty + h
	
	# Get possible x-text by moving from where we
	# left off in x-labels to the complete
	# height of the image.
	maxIntersection = 0
	maxList = []
	for i in range(hmax + 1, height):
		count = 0
		current = []
		for index, (text, rect) in enumerate(x_text):
			if lineIntersectsRectY(i, rect):
				count += 1
				current.append(x_text[index])
							
		if count > maxIntersection:
			maxIntersection = count
			maxList = current
	
	x_text = []
	for text, (textx, texty, w, h) in maxList:
		x_text.append(text)
	
	# Get possible legend text
	# For this, we need to search both top to
	# bottom and also from left to right.
	
	legends_and_numbers = mergeTextBoxes(legends)
		
	legends = []
	for text, (textx, texty, w, h) in legends_and_numbers:
		if not re.search(r'^([(+-]*?(\d+)?(?:\.\d+)*?[-%) ]*?)*$', text):
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
	
	#print("\n\n")
	
	maxList = []
	
	if len(legend_groups) > 0:
		maxList = max(legend_groups, key = len)
		
	legends = []
	for text, (textx, texty, w, h) in maxList:
		legends.append(text)
			
	return image, x_labels, x_labels_list, x_text, y_labels, y_labels_list, y_text_list, legends, maxList

# Getting the Ratio for y-value matching
# Similar to the label detection logic, y-ticks are detected:
# Y-ticks

# Check only the numerical boxes which are to the left of y-axis and to the top of x-axis.
# Run a line sweep from left end of the image to the y-axis position, and check when the sweeping line intersects with the maximum number of numerical boxes.
# The numerical boxes are then used as bounding boxes for calculating the y-ticks.
#Difference between the y-ticks is then calculated.
#Only consider the mean difference between the y-ticks, rejecting the outliers from the calculated values.
#Finally, the value-tick ratio to normalize the heights of the bounding boxes is calculated by:


def getRatio(path, image_text, xaxis, yaxis):
	list_text = []
	list_ticks = []
	
	filepath = path
	
	image = cv2.imread(filepath)
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
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
	return text_sorted, normalize_ratio



# def reject_outliers(data, m=1):
# 	return data[abs(data - np.mean(data)) <= m * np.std(data)]

#This version base on percentiles is better:
def reject_outliers(data, lower_percentile=5, upper_percentile=95):
	lower_bound = np.percentile(data, lower_percentile)
	upper_bound = np.percentile(data, upper_percentile)
	return data[(data >= lower_bound) & (data <= upper_bound)]


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
	image = cv2.imread(filepath)
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
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
	image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
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




# Matching the ratio for final data extraction
# Y-val data: - The height of each bounding box is recorded by the help of the merging 
#rectangles during Cluster count estimation method. - Eventually, we used the ratio to 
#calculate the y-values as:

def mergeRects(contours, mode='contours'):
	rects = []
	rectsUsed = []
	
	# Just initialize bounding rects and set all bools to false
	for cnt in contours:
		if mode == 'contours':
			rects.append(cv2.boundingRect(cnt))
		elif mode == 'rects':
			rects.append(cnt)
		
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
	
	return acceptedRects



def mergeTextBoxes(textboxes):
	rects = []
	rectsUsed = []
	
	# Just initialize bounding rects and set all bools to false
	for box in textboxes:
		rects.append(box)
		rectsUsed.append(False)
	# Sort bounding rects by x coordinate
	def getXFromRect(item):
		return item[1][0]
	
	def getYFromRect(item):
		return item[1][1]
	rects.sort(key = lambda x: (getYFromRect, getXFromRect))
	
	# Array of accepted rects
	acceptedRects = []
	# Merge threshold for x coordinate distance
	xThr = 10
	yThr = 0
	# Iterate all initial bounding rects
	for supIdx, supVal in enumerate(rects):
		if (rectsUsed[supIdx] == False):
			# Initialize current rect
			currxMin = supVal[1][0]
			currxMax = supVal[1][0] + supVal[1][2]
			curryMin = supVal[1][1]
			curryMax = supVal[1][1] + supVal[1][3]
			currText = supVal[0]
			# This bounding rect is used
			rectsUsed[supIdx] = True
			# Iterate all initial bounding rects
			# starting from the next
			for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):
				# Initialize merge candidate
				candxMin = subVal[1][0]
				candxMax = subVal[1][0] + subVal[1][2]
				candyMin = subVal[1][1]
				candyMax = subVal[1][1] + subVal[1][3]
				candText = subVal[0]
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
					currText = currText + ' ' + candText
					
					# Merge candidate (bounding rect) is used
					rectsUsed[subIdx] = True
				else:
					break
			# No more merge candidates possible, accept current rect
			acceptedRects.append([currText, (currxMin, curryMin, currxMax - currxMin, curryMax - curryMin)])
	
	return acceptedRects



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


def euclidean(v1, v2):
	return sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5


def angle_between(p1, p2):
	deltaX = p1[0] - p2[0]
	deltaY = p1[1] - p2[1]
	return math.atan2(deltaY, deltaX) / math.pi * 180
   

def RectDist(rectA, rectB):
	(rectAx, rectAy, rectAw, rectAh) = rectA
	(rectBx, rectBy, rectBw, rectBh) = rectB
	
	return abs(rectAx + rectAw / 2 - rectBx - rectBw / 2)

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


def expand(points, margin):
	return np.array([
		[[points[0][0][0] - margin, points[0][0][1] - margin]],
		[[points[1][0][0] + margin, points[1][0][1] - margin]],
		[[points[2][0][0] + margin, points[2][0][1] + margin]],
		[[points[3][0][0] - margin, points[3][0][1] + margin]]])
	

def filterBbox(rects, legendBox):
	text, (textx, texty, width, height) = legendBox
	
	filtered = []
	for rect in rects:
		(x, y, w, h) = rect
		if abs(y - texty) <= 10 and abs(y - texty + h - height) <= 10:
			filtered.append(rect)
	filtered = mergeRects(filtered, 'rects')
	
	closest = None
	dist = sys.maxsize
	for rect in filtered:
		(x, y, w, h) = rect
		if abs(x + w - textx) <= dist:
			dist = abs(x + w - textx)
			closest = rect
	
	return closest



def boxGroup(img, box):
	(x, y, w, h) = box
	image = img[y:y+h, x:x+w].reshape((h * w, 3))
	values, counts = np.unique(image, axis = 0, return_counts = True)
	
	# Remove white and near-by pixels
	threshold = 5
	for r in range(255 - threshold, 256):
		for g in range(255 - threshold, 256):
			for b in range(255 - threshold, 256):
				image = image[np.where((image != [r, g, b]).any(axis = 1))]
	values, counts = np.unique(image, axis = 0, return_counts = True)
				
	sort_indices = np.argsort(-counts)
	values, counts = values[sort_indices], counts[sort_indices]
	groups = []
	groupcounts = []
	for idx, value in enumerate(values):
		grouped = False
		for groupid, group in enumerate(groups):
			for member in group:
				r, g, b = member
				vr, vg, vb = value
				if (abs(vr.astype(np.int16) - r.astype(np.int16)) <= 5 and
					abs(vg.astype(np.int16) - g.astype(np.int16)) <= 5 and
					abs(vb.astype(np.int16) - b.astype(np.int16)) <= 5):
						group.append(value)
						groupcounts[groupid] += counts[idx]
						grouped = True
						break
			if grouped:
				break
		if not grouped:
			groups.append([value])
			groupcounts.append(counts[idx])
	groupcounts = np.array(groupcounts)
	sort_indices = np.argsort(-groupcounts)
	new_groups = [groups[i] for i in sort_indices]
	groups = new_groups
	
	return groups


#Saving y-values in our data excel sheet
def getYVal(index, path, yValueDict, image_text, texts, image_extensions):
	filepath = path
	img = cv2.imread(filepath)
	img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_height, img_width, _ = img.shape
	
	# Axes detection
	xaxis, yaxis = detectAxes(filepath)
	
	for (x1, y1, x2, y2) in [xaxis]:
		xaxis = (x1, y1, x2, y2)
		
	for (x1, y1, x2, y2) in [yaxis]:
		yaxis = (x1, y1, x2, y2)
		
	img, x_labels, x_labels_list, _, _, _, _, legends, legendBoxes = getProbableLabels(img, image_text, xaxis, yaxis)
	actual_image = img.copy()
	
	try:
		list_text, normalize_ratio = getRatio(path, image_text, xaxis, yaxis)
		print("[getYVal] legends: {0}".format(legends))
		print("[{0}] path: {1}, ratio: {2}".format(index, path.name, normalize_ratio), end='\n\n')
		
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
				
		for i in range(len(groups)):
			img = cv2.imread(filepath)
			img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
		
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
			if len(contours) == 0 or len(contours) > 100:
				continue
			contours = [cv2.approxPolyDP(contour, 3, True) for contour in contours]
			rects = mergeRects(contours)
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
			y_val = [(rect, round(l* normalize_ratio, 1)) for rect, l in list_len]
			
			for x_label, box in x_labels_list:
				(x, y, w, h) = box
				value = 0.0
				closest = None
				dist = sys.maxsize
				
				for index, item in enumerate(y_val):
					if labels[index] == x_label:
						(vx, vy, vw, vh) = item[0]
						if abs(x + w/2 - vx - vw/2) < dist:
							dist = abs(x + w/2 - vx - vw/2)
							closest = item[0]
							value = item[1]
					 
				data[legendtext][x_label] = value
			 
		yValueDict[path.name] = data
		
	except Exception as e:
		print(e)
		return
					
	return yValueDict



