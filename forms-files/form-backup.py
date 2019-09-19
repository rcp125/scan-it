from cv2 import cv2
import numpy as np
import pytesseract
from PIL import Image
from os import walk

path = "C://Users//HHS//Desktop//ScanIt//"
# imgPath = path + "images//tableintable_a1.jpg"

f = []
for (dirpath, dirnames, filenames) in walk('C:/Users/HHS/Desktop/ScanIt/forms'):
    f.extend(filenames)
    break


def removeLines(path, imgPath):
    img = cv2.imread(imgPath, 0)
    img = 255 - img
    _, thres = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path + "removed//thres.jpg", thres)

    horizontal = thres
    vertical = thres
    cols = horizontal.shape[1]
    h_kernel = cols // 30

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel,1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    cv2.imwrite(path + "removed//horizontal.jpg", horizontal)

    rows = vertical.shape[0]
    v_kernel = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    cv2.imwrite(path + "removed//vertical.jpg", vertical)

    _, edges = cv2.threshold(vertical,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path + "removed//edges.jpg", edges)

    kernel = np.ones((2, 2), dtype = "uint8")
    dilated = cv2.dilate(edges, kernel)
    cv2.imwrite(path + "removed//dilated.jpg", dilated)

    mask = cv2.bitwise_not(vertical+horizontal)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    cv2.imwrite(path + "removed//invh.jpg", mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img_inv = cv2.bitwise_not(masked_img)

    cv2.imwrite(path + "removed_result.jpg", masked_img_inv)

    return masked_img_inv


def preprocess(path, imgPath):
    gray = cv2.imread(path+"removed_result.jpg", 0)
    inverted = 255 - gray
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(inverted, kernel)
    cv2.imwrite(path+"sample//dilated.jpg",dilated)

    _, thres = cv2.threshold(dilated, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(path+"sample//threshold.jpg",thres)
    
    kernel = np.ones((1,15), np.uint8)
    blobs = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(path+"sample//blobs.jpg",blobs)
    
    return blobs

def identifyChunks(imgPath, blobs, i, RECTpadding=2, OCRpadding = 3):
    i = i + 1
    img = cv2.imread(imgPath)
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
                cv2.rectangle(img, (x-RECTpadding, y-RECTpadding), (x+w+RECTpadding, y+h+RECTpadding), (0, 0, 255), 2)

                # while True:
                #     try:
                #         textExtract(img[y-OCRpadding:y+h+OCRpadding, x-OCRpadding:x+w+OCRpadding], i)
                #         break
                #     except:
                #         OCRpadding = OCRpadding - 1
                
    cv2.imwrite(path+"sample//" + str(i) + "result.jpg",img)

    return img

def textExtract(img, i):
    img = Image.fromarray(img, 'RGB')
    # img.show()
    # cv2.waitKey(0)
    loc = "C://Users//HHS//Desktop//ScanIt//sample//ocr" + str(i) + ".jpg"

    img.save(loc)
    img = Image.open(loc).convert('L')
    _,img = cv2.threshold(np.array(img), 125, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(img)
    print(text)
    if(text == ""):
        imgArray.append(img)
    else:
        textArray.append(text)

if __name__ == '__main__':
    textArray = []
    imgArray = []

    k=0

    for i in f:
        img = removeLines(path, path+"images/"+i)
        blobs = preprocess(path, path+"images/"+i)
        img = identifyChunks(path+"images/"+i, blobs, k)
    
    # print(textArray)
    # print(imgArray)
