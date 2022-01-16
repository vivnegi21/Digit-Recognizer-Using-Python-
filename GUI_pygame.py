#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Importing All the libraries
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pygame
import matplotlib.pyplot as plt


# In[4]:



#initializing pygame
pygame.init()

#Setting the display size and color
screen = pygame.display.set_mode((300, 400))
screen.fill((255, 255, 255))
pygame.display.set_caption("Write a digit")
screen.get_size()


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
img2=cv2.imread('num.png',0)
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

cv2.putText(img2,str(np.argmax(pred[0])),(0,390),cv2.FONT_HERSHEY_COMPLEX, 2 , (0,255,255) ,3)
cv2.imshow("Predictions", img2)
cv2.waitKey()

    


# In[ ]:





# In[ ]:




