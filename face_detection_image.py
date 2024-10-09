import numpy as np
import cv2
import matplotlib.pyplot as plt

# Reading the image 
image = cv2.imread('./cast2n1d.jpg')

if image is None:
    print("Error: Image not found or unable to read.")
else:
    print("Image loaded successfully.")

# Cho hinh anh lon hon
scale_factor = 2
width = int(image.shape[1] * scale_factor)
height = int(image.shape[0] * scale_factor)
dim = (width, height)

# 
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

# Converting image to grayscale 
gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY) 

# Loading the required haar-cascade xml classifier file 
haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Applying the face detection method on the grayscale image 
faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9) 

# Iterating through rectangles of detected faces 
for (x, y, w, h) in faces_rect: 
	cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2) 

# Display the resized image with detected faces
cv2.imshow('Detected faces', resized_image) 

cv2.waitKey(0)
