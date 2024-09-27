"""
This code gives a best estimate of the x and y axis 
(horizontal and vertical axes) for the plot/chart.

Based on https://github.com/Cvrane/ChartReader/blob/master/code/AxesDetection.ipynb
"""

import cv2, imutils
import numpy as np

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
	#image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
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
