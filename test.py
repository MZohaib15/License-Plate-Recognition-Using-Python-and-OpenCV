import cv2

from PIL import Image
import pytesseract as tess

image = cv2.imread('t4.jpg')

text=tess.image_to_string(image)
print(text)

