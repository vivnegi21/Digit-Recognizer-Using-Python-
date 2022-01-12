#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing All the libraries
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pygame
import matplotlib.pyplot as plt


# In[8]:


#initializing pygame
pygame.init()

#Setting the display size and color
screen = pygame.display.set_mode((300, 400))
screen.fill((255, 255, 255))
pygame.display.set_caption("Write a digit")



#To create a paint like canvas
loop = True
while loop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #Saving the image as num.png
            
            pygame.image.save(screen, 'num.png')
            loop = False
    x, y = pygame.mouse.get_pos()
    if pygame.mouse.get_pressed() == (1, 0, 0):
        pygame.draw.circle(screen, (0, 0, 0), (x, y), 20)
    pygame.display.update()

#Quitting the pygame terminal
pygame.quit()

# reading the image made earlier 
img = cv2.imread("num.png")

#calling the model trained
model=keras.models.load_model("model/mnist_p.h5")

img=cv2.imread('num.png',0)

#reversing the colors of img
img=cv2.bitwise_not(img)

#resizing and reshaping the img variable
img=cv2.resize(img,(28,28))
img=img.reshape(1,28,28,1)

#normalizing the image
img=img/255.

#making a prediction
pred=model.predict(img)

#making the im variable for image again to make the boundary on number
im = cv2.imread("num.png")

#converting the image to the way of images we trained our data(but in inverse colors)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)
#use this to see the image made
#plt.imshow(im_gray)

#revering the colors
im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)[1]
#use this to see the thermal-like image
#plt.imshow(im_th)

#to create a green rectangle
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    
    height, width = roi.shape
    
    if height != 0 and width != 0:
        cv2.putText(im, str(np.argmax(pred[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
cv2.imshow("Predictions", im)
cv2.waitKey()


# In[ ]:




