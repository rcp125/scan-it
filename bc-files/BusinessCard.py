from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

path = "C://Users//HHS//Desktop//ScanIt//"
imgPath = path + "cards//repo//card.jpg"
img = cv2.imread(imgPath)

class CardDetails():
    def __init__(self, firstname, lastname, prefix, company, title, pic, phone, fax, address, email):
        self.firstname = firstname
        self.lastname = lastname
        self.prefix = prefix
        self.company = company
        self.title = title
        self.pic = pic
        self.phone = phone
        self.fax = fax
        self.address = address
        self.email = email


def preprocess(path, imgPath):
    gray = cv2.imread(imgPath, 0)
    inverted = 255 - gray
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    cv2.imwrite(path+"sample//dilated.jpg",dilated)

    ret, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path+"sample//threshold.jpg",thres)
    
    kernel = np.ones((1,10), np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(path+"sample//blobs.jpg",blobs)
    
    return blobs

def identifyChunks(blobs, padding=2):
    contours, hierarchy = cv2.findContours(blobs.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # create mask layer with black background
    mask = np.zeros(blobs.shape, np.uint8)
    
    for contour in range(len(contours)):
            (x, y, w, h) = cv2.boundingRect(contours[contour])

            # mask around bounding rectangle
            mask[y:y+h, x:x+w] = 0
            
            # white fill to bounding rectangle area
            cv2.drawContours(mask, contours, contour, (255, 255, 255), -1)
            
            # calculate ratio of non-black
            ratio = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
            
            if ratio > 0.5 and w > 5 and h > 5:
                # cv2.rectangle(img, (x-padding, y-padding), (x+w+padding, y+h+padding), (0, 0, 255))
                textExtract(img[y-3:y+h+3, x-3:x+w+3])
                
    cv2.imwrite(path+"sample//result.jpg",img)

    return img

def textExtract(img):
    img = Image.fromarray(img, 'RGB')
    # img.show()
    # cv2.waitKey(0)
    text = pytesseract.image_to_string(img)
    if(text == ""):
        imgArray.append(img)
    else:
        textArray.append(text)


def interpret(field, cellWords, email):
    # if re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", field):
    #     return("e")
    if re.match(r"^\s*(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?\s*$", field) or any(substring in field for substring in cellWords):
        return("c")
    
    if (field.split()[0].lower()) in email.lower():
        return("n")

    
    

def formatCell(field):
    return(re.sub(r"\D", "", field))

def formatName(names):
    for name in names:
        tempNameArray = name.split()
        if(len(tempNameArray) > 1):
            return name

if __name__ == '__main__':
    textArray = []
    imgArray = []
    email = ""
    names = []
    cellWords = ["C:", "Cell", "Cell Phone", "Cell:"]
    cell = ""


    blobs = preprocess(path, imgPath)
    img = identifyChunks(blobs)
    
    fields = textArray
    # logo = imgArray[0].show()

    # moves email to front of array
    for field in fields:
        if re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", field):
            fields.remove(field)
            email = field

    for field in fields:
        val = interpret(field, cellWords, email)
        if(val == "c"):
            cell = formatCell(field)
        if(val == "n"):
            names.append(field)
        
    name = formatName(names)   

    print(name)
    print(cell)
    print(email)     
        

