import cv2
import numpy as np
import Utils

# Load the image
path = '1.png'
width = 500
height = 900    
image = cv2.imread(path)

# Preprocess the image
image = cv2.resize(image, (width, height))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(thresh, 50, 150)
contourImg = image.copy()

# Find all of the contours
contour, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contourImg, contour, -1, (0, 255, 0), 1)

# Find rectangles
rectCon = Utils.rectContour(contour)
if len(rectCon) > 0:
    BiggestRectCon = Utils.getCornerPoints(rectCon[0])
    print(BiggestRectCon)

    if BiggestRectCon.size != 0:
        cv2.drawContours(contourImg, [BiggestRectCon], -1, (255, 0, 0), 2)

imageBlank = np.zeros_like(image)
imageArray = ([
                image,
                gray,
                blur,
                thresh,
                canny,
                contourImg,
                # imageBlank
               ])
stackedImage = Utils.stackImages(imageArray, 0.5)

# Display the stacked images
cv2.imshow('Stacked Images', stackedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()