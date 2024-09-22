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