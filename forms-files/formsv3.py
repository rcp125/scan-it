from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd

'''
Function Notes:

1. removeLines:
    - attempts to remove vertical & horizontal lines from forms
      so they do not interfere with blob creation
    - significantly improves rectangle accuracy
    - may not work if the line thickness is greater than "normal"

2. preprocess
    - image manipulations to create blobs that surround isolated text

3. identifyChunks
    - finds contours of text chunks

4. textType
    - determines whether chunk containers printed or handwritten text

5. signatureExtract
    - attempts to detect which chunk contains a signature and returns image

6. textExtract
    - performs OCR on text blocks

'''

path = "C://Users//HHS//Desktop//TableFinder//"
imgPath = path + "forms//connected.jpg"

ocrPerformed = False

def removeLines():
    gray = cv2.imread(imgPath, 0)
    inverted = cv2.bitwise_not(gray)
    _, thres = cv2.threshold(inverted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path + "removeLines//thres.jpg", thres)

    horizontal = thres
    vertical = thres
    cols = horizontal.shape[1]
    h_kernel = cols // 30

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel,1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite(path + "removeLines//horizontal.jpg", horizontal)

    rows = vertical.shape[0]
    v_kernel = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    kernel = np.ones((2, 2), dtype = "uint8")
    dilated = cv2.dilate(vertical, kernel)
    cv2.imwrite(path + "removeLines//vertical.jpg", dilated)

    mask = cv2.bitwise_not(vertical+horizontal)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    cv2.imwrite(path + "removeLines//frame.jpg", mask)

    masked_img = cv2.bitwise_and(inverted, inverted, mask=mask)
    removed_result = cv2.bitwise_not(masked_img)
    cv2.imwrite(path + "removeLines//removed_result.jpg", removed_result)

    return removed_result
    

def preprocess(removed_result, blob_kernel=(1,15)):
    inverted = cv2.bitwise_not(removed_result)
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    cv2.imwrite(path+"preprocess//dilated.jpg",dilated)

    _, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path+"preprocess//threshold.jpg",thres)
    
    kernel = np.ones(blob_kernel, np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(path+"preprocess//blobs.jpg",blobs)
    
    return blobs

def identifyChunks(blobs, RECTpadding=2, OCRpadding = 3, drawRect = True, ocr = False, textType = False):
    regions = []    # index, coordinates, density
    contours, _ = cv2.findContours(blobs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(blobs.shape, np.uint8)
    
    textType = ""
    text = ""
    i = 0
    for contour in range(len(contours)):
            i = i + 1
            (x, y, w, h) = cv2.boundingRect(contours[contour])
            mask[y:y+h, x:x+w] = 0  # mask around bounding rectangle
            cv2.drawContours(mask, contours, contour, (255, 255, 255), -1)  # white fill to bounding rectangle area
            # cv2.imwrite(path+"sample//result" + str(i) + ".jpg",mask)
            white = np.sum(mask[y:y+h, x:x+w] == 255)
            density = float (white / (w*h))
            regions.append([i, (x,y,w,h), density, text, textType])
    return regions

def drawRect(regions, padding=2):
    rect = cv2.imread(imgPath)
    for region in regions:
        (x, y, w, h) = region[1]
        if w > 5 and h > 5:
            cv2.rectangle(rect, (x-padding, y-padding), (x+w+padding, y+h+padding), (0, 0, 255), 2)
    
    cv2.imwrite(path+"results//result.jpg", rect)

def ocr(regions, padding = 2):  # slow, ~30s runtime for one form
    global ocrPerformed
    ocrPerformed = True
    img = cv2.imread(imgPath)

    for region in regions:
        i = region[0]
        (x, y, w, h) = region[1]
        
        snippetArray = img[y-padding:y+h+padding, x-padding:x+w+padding]

        snippet = Image.fromarray(snippetArray, 'RGB')
        loc = path + "ocr//" + str(i) + ".jpg"
        snippet.save(loc)

        snippet = Image.open(loc).convert('L')
        _,snippet = cv2.threshold(np.array(snippet), 125, 255, cv2.THRESH_BINARY)
        
        text = pytesseract.image_to_string(snippet)

        df = pytesseract.image_to_data(snippet, output_type='data.frame')
        filtered = df[df.conf != -1]
        confidence_series = filtered.groupby(['block_num'])['conf'].mean()
        try:
            confidence_val = confidence_series.iloc[0]
        except:
            confidence_val = 0

        if(confidence_val < 50):
            region[4] == "handwritten"

        region[3] = text

        print(str(i) + ". " + str(confidence_val) + " : " + region[3])
        
def textType(regions, padding = 2):
    if(ocrPerformed == False):
        ocr(regions)
    for region in regions:
        if(region[3] == ""):
            region[4] = "handwritten"
        else:
            region[4] = "printed"
    
    for region in regions:
        print(str(region[0]) + ". " + region[4] + " :" + region[3])

if __name__ == '__main__':
    removed_result = removeLines()
    blobs = preprocess(removed_result)
    regions = identifyChunks(blobs)
    drawRect(regions)
    # ocr(regions)
    # textType(regions)