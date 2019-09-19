from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image


path = "C://Users//HHS//Desktop//ScanIt//"
imgPath = path + "itinerary//source.jpg"

def preprocess(path, imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path+"itinerary//gray.jpg", gray)
    inverted = 255 - gray
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    _, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((1,12), np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(path+"itinerary//blobs.jpg",blobs)
    return blobs

def identifyChunks(imgPath, blobs, RECTpadding=2, OCRpadding = 3, drawRect = True, ocr = False):
    img = cv2.imread(imgPath)
    rect = img.copy()
    contours, _ = cv2.findContours(blobs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # create mask layer with black background
    mask = np.zeros(blobs.shape, np.uint8)
    
    i = 0
    for contour in range(len(contours)):
            i = i + 1
            (x, y, w, h) = cv2.boundingRect(contours[contour])

            # mask around bounding rectangle
            mask[y:y+h, x:x+w] = 0
            
            # white fill to bounding rectangle area
            cv2.drawContours(mask, contours, contour, (255, 255, 255), -1)
            # cv2.imwrite(path+"sample//result" + str(i) + ".jpg",mask)

            
            # calculate ratio of non-black
            ratio = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
            
            if  w > 5 and h > 5:
                if(drawRect == True):
                    cv2.rectangle(rect, (x-RECTpadding, y-RECTpadding), (x+w+RECTpadding, y+h+RECTpadding), (0, 0, 255), 2)

                if(ocr == True):
                    while True:
                        try:
                            textExtract(img[y-OCRpadding:y+h+OCRpadding, x-OCRpadding:x+w+OCRpadding], i)
                            break
                        except:
                            OCRpadding = OCRpadding - 1
                
    cv2.imwrite(path+"itinerary//result.jpg",rect)

    return img



def ocr(img, i):
    text = pytesseract.image_to_string(img)
    print(text)
    

if __name__ == "__main__":
    blobs = preprocess(path, imgPath)
    img = identifyChunks(imgPath, blobs)