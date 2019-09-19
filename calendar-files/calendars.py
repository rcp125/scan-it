from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image

'''

####################################
#######      INCOMPLETE      #######
####################################

'''

path = "C://Users//HHS//Desktop//Projects//scanit//calendar-files//"
imgPath = path + "itinerary//source.jpg"

def preprocess(path, imgPath):
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path+"itinerary//gray.jpg", gray)
    inverted = 255 - gray
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    _, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)kernel = np.ones((1,15), np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(path+"itinerary//blobs.jpg",blobs)
    return blobs




def ocr(img, i):
    text = pytesseract.image_to_string(img)
    print(text)
    

if __name__ == "__main__":
    preprocess(path, imgPath)