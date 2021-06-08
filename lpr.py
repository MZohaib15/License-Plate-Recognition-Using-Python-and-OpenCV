#!/usr/bin/env python
import numpy as np
import cv2
import  imutils
from PIL import Image
import pytesseract as tess

image = cv2.imread('t9.jpg')

image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image Grayscale Version", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Image Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edge Detection", edged)

cntrs,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cntrs=sorted(cntrs, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCntr = None 

count = 0
for c in cntrs:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            NumberPlateCntr = approx 
            break

cv2.drawContours(image, [NumberPlateCntr], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)


def transform(pos):
    pts = []
    n = len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums = {}
    diffs = {}
    tl = tr = bl = br = 0
    for i in pts:
        x = i[0]
        y = i[1]
        sum = x + y
        diff = y - x
        sums[sum] = i
        diffs[diff] = i
    sums = sorted(sums.items())
    diffs = sorted(diffs.items())
    n = len(sums)
    rect = [sums[0][1], diffs[0][1], diffs[n - 1][1], sums[n - 1][1]]
    h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  
    h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  
    h = max(h1, h2)
    w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  
    w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2) 
    w = max(w1, w2)

    return int(w), int(h), rect

n = len(cntrs)
max_area = 0
pos = 0
for i in cntrs:
    area = cv2.contourArea(i)
    if area > max_area:
        max_area = area
        pos = i
perimeter = cv2.arcLength(pos, True)
approx = cv2.approxPolyDP(pos, 0.02 * perimeter, True)

size = image.shape
w, h, arr = transform(approx)

pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts1 = np.float32(arr)
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(image, M, (w, h))
image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
cv2.imshow('OUTPUT', image)

text=tess.image_to_string(image)
print(text)

cv2.waitKey(0)


