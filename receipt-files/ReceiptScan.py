from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image


path = "C://Users//HHS//Desktop//ScanIt//Receipts//"
imgPath = path + "repo//" + "Receipt1.jpg"
ocrPerformed = False

class TextChunk():
    def __init__(self, index, coordinates, density, text, row):
        self.index = index
        self.coordinates = coordinates
        self.density = density
        self.text = text
        self.row = row

    def getXY(self):
        return (self.coordinates[0], self.coordinates[1])
    
    def getX2(self):
        return (self.coordinates[0] + self.coordinates[2])

    def getY2(self):
        return(self.coordinates[1] + self.coordinates[3])

def threshold(gray, whiteBG = False):
    if(whiteBG == False):
        _, thres = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        thres = cv2.erode(thres, kernel)
    
    else:
        inverted = cv2.bitwise_not(gray)
        _, thres = cv2.threshold(inverted,40,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        thres = cv2.erode(thres, kernel)

    cv2.imwrite(path + "thres.jpg", thres)
    return thres

def find_contours(thres):
    contours, _ = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        return cv2.boundingRect(c)

def outermost(RECTpadding = 0):
    img = cv2.imread(imgPath)
    rect = img.copy()
    gray = cv2.imread(imgPath, 0)

    thres = threshold(gray)
    (x,y,w,h) = find_contours(thres)

    if(w*h == gray.size):
        thres = threshold(gray, True)
        (x,y,w,h) = find_contours(thres)

    cv2.rectangle(rect, (x-RECTpadding, y-RECTpadding), (x+w+RECTpadding, y+h+RECTpadding), (0, 0, 255), 2)
    cv2.imwrite(path + "rect.jpg", rect)

    snippet = img[y:y+h, x:x+w]
    cv2.imwrite(path+"snippet.jpg", snippet)
    return snippet

def preprocess(snippet):
    gray = cv2.imread(path+"snippet.jpg", 0)
    inverted = cv2.bitwise_not(gray)
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted,kernel)
    cv2.imwrite(path+"dilated.jpg",dilated)

    _, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path+"threshold.jpg",thres)
    
    kernel = np.ones((1,40), np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(path+"blobs.jpg",blobs)
    
    return blobs

def identifyChunks(blobs, padding=2):
    regions = []    # index, coordinates, density
    contours, _ = cv2.findContours(blobs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(blobs.shape, np.uint8)
    
    text = ""
    row = None

    i = 0
    for contour in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[contour])
        if (w>10 and h>10) and (h<300):
            i = i + 1
            mask[y:y+h, x:x+w] = 0  # mask around bounding rectangle
            cv2.drawContours(mask, contours, contour, (255, 255, 255), -1)  # white fill to bounding rectangle area
            # cv2.imwrite(path+"sample//result" + str(i) + ".jpg",mask)
            white = np.sum(mask[y:y+h, x:x+w] == 255)
            density = float (white / (w*h))
            obj = TextChunk(i, (x,y,w,h), density, text, row)
            regions.append(obj)
    return regions

def cluster(regions):
    j=0
    for i in range(len(regions) - 1):
        lo = regions[i][1][1]
        if(lo - 15 < regions[i+1][1][1] < lo + 15):
            print(j)
            j=j+1

def drawRect(regions, padding_x=2, padding_y = 2):
    rect = cv2.imread(path+"snippet.jpg")
    for region in regions:
        (x, y, w, h) = region.coordinates
        cv2.rectangle(rect, (x-padding_x, y-padding_y), (x+w+padding_x, y+h+padding_y), (0, 0, 255), 2)
        cv2.putText(rect, str(region.index), (region.coordinates[0], region.coordinates[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    cv2.imwrite(path+"result.jpg", rect)
    

def ocr(regions, padding = 2):  # slow, ~30s runtime for one form
    global ocrPerformed
    ocrPerformed = True
    img = cv2.imread(path + "snippet.jpg")

    for region in regions:
        i = region.index
        print(i)
        (x, y, w, h) = region.coordinates
        
        snippetArray = img[y-padding:y+h+padding, x-padding:x+w+padding]

        snippet = Image.fromarray(snippetArray, 'RGB')
        loc = path + "ocr//" + str(i) + ".jpg"
        snippet.save(loc)

        snippet = Image.open(loc).convert('L')
        _,snippet = cv2.threshold(np.array(snippet), 125, 255, cv2.THRESH_BINARY)
        
        text = pytesseract.image_to_string(snippet, config="--psm 13")

        df = pytesseract.image_to_data(snippet, output_type='data.frame')
        filtered = df[df.conf != -1]
        confidence_series = filtered.groupby(['block_num'])['conf'].mean()
        try:
            confidence_val = confidence_series.iloc[0]
        except:
            confidence_val = 0

        # if(confidence_val < 50):
        #     continue

        region.text = text

        # print(str(i) + ". " + str(confidence_val) + " : " + region.text)

        textArray.append(region.text)

def ocrWhole():
    img = cv2.imread(imgPath)
    text = pytesseract.image_to_string(img)
    print(text)


def checkEditDistance(str1, str2):
    if(str1 == str2):
        return True

    len1 = len(str1)
    len2 = len(str2)

    if ((len1 - len2) > 1 or (len2 - len1) > 1  ):
        return False

    i = 0
    j = 0
    diff = 0
    
    while (i<len1 and j<len2): 
        f = str1[i]
        s = str2[j]
        if (f != s):
            diff = diff + 1
            if (len1 > len2):
                i = i + 1
            if (len2 > len1):
                j = j + 1
            if (len1 == len2):
                i = i + 1
                j = j + 1
        else:
            i = i + 1
            j = j + 1

        if (diff > 1):
            return False

    if (diff == 1 and len1 != len2 and (i != len1 or j != len2)):
        return False

    return True

if __name__ == '__main__':
    textArray = []
    snippet = outermost()
    blobs = preprocess(snippet)
    regions = identifyChunks(blobs)
    ocr(regions)

    # ocrWhole()

    for text in textArray:
        print(checkEditDistance(text, "Total"))  