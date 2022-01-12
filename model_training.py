#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing All necessary modules for model_training
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


# fetching data set from keras
(X_train_full,Y_train_full),(X_test,Y_test) = keras.datasets.mnist.load_data()


# In[3]:



# Showing the shapes of tables be used

print(f"The shape of X_train_full is {X_train_full.shape}")
print(f"The shape of Y_train_full is {Y_train_full.shape}")
print(f"The shape of X_test is {X_test.shape}")
print(f"The shape of Y_test is {Y_test.shape}")


# ## The images in the data is 28 X 28 in pixels with intensity between 0 and 255

# In[4]:


# Viewing the data for first row of X_train_full
X_train_full[0]


# In[5]:


# Seeing images from rows and its true value
plt.rcParams['figure.figsize'] = (15, 5)
plt.subplot(1,5,1)
plt.imshow(X_train_full[0])
plt.title(Y_train_full[0])


plt.subplot(1,5,2)
plt.imshow(X_train_full[1])
plt.title(Y_train_full[1])

plt.subplot(1,5,3)
plt.imshow(X_train_full[2])
plt.title(Y_train_full[2])

plt.subplot(1,5,4)
plt.imshow(X_train_full[3])
plt.title(Y_train_full[3])

plt.subplot(1,5,5)
plt.imshow(X_train_full[4])
plt.title(Y_train_full[4])


# In[6]:


# declaring the class names
# here class is used because the output neuron will give 10 class respectively.

class_names=[0,1,2,3,4,5,6,7,8,9]


# # Data Normalization
# * We then normalize the data dimensions so that they are of approximately the same size.

# In[7]:


X_train_full = X_train_full.reshape(X_train_full.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# normalizing The Data
X_train_n = X_train_full/255.
X_test_n = X_test / 255. 


# ## Split the data into train/vadilation /test datasets
# In the earlier step of importing the data, we had 60000 datasets for training data into train/validation.
# Here is how each datasets is used in deep learning.
# * Training data : used for the training model
# * Validation data : used for tuning the hyperparameters and evaluate the model
# * Test data : used to test the model after the model has gone through the initial vetting by the validation set.
# 

# In[8]:


X_valid,X_train = X_train_n[:5000],X_train_n[5000:]
Y_valid, Y_train = Y_train_full[:5000],Y_train_full[5000:]
X_test = X_test_n


# # Data is prepared to be used in Model Training

# .

# .

# .

# .

# # Create a  model architecture
# We'll use sequential model to classify handwritten digits.

# .

# .

# .

# In[9]:


# making random seed to make same result come everytime we fit the model with same data
np.random.seed(42)
tf.random.set_seed(42) 

# used to replicate same result everytime


# In[10]:


batch_size = 128
num_classes = 10
epochs = 10
# model making

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3,3),strides= (1,1),padding="valid",activation = "relu",input_shape=[28,28,1]))
model.add(keras.layers.Conv2D(64, (3,3),activation = "relu",strides = (1,1),padding="valid"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3,3),activation = "relu",strides = (1,1),padding="valid"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))


# In[11]:


# Model Architecture
model.summary()


# In[12]:


# model shape
keras.utils.plot_model(model, "model_with_shape_info.png", show_shapes=True)


# In[13]:


# To show the weights we use .get_weights()

#model.get_weights()

# these are random weights for now we will use SGD(optimization technique) to get minimum of function


# In[14]:


# to get weights and biases
weights , biases = model.layers[1].get_weights()


# In[15]:


weights.shape


# In[16]:


# compiling the model 
# here cross entropy error function is used and stochastic Gradient Descent as activation function
model.compile(loss="sparse_categorical_crossentropy",
             optimizer = "sgd",
             metrics=["accuracy"]
             )


# In[17]:


# Creating callbacks and saving model during training
checkpoint_cb = keras.callbacks.ModelCheckpoint("Best_model.h5",save_best_only=True)
early_stop_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)


# In[18]:


# moodel fitting with training and validation data
hist = model.fit(X_train, Y_train,batch_size=batch_size,epochs=100,verbose=1,validation_data=(X_test, Y_test),callbacks=[checkpoint_cb,early_stop_cb])


# In[19]:


hist.params


# In[20]:


# to know all the epoch results use this

hist.history


# In[21]:


# plotting the loss and accuracy of training and validation data
pd.DataFrame(hist.history).plot()
plt.grid()
plt.show()


# In[22]:


# to get the training accuracy
model.evaluate(X_train,Y_train)


# In[23]:


# to get the testing accuracy
model.evaluate(X_test,Y_test)


# .

# # Testing data on testing data

# In[24]:


X_new=X_test[:5]
print(f"the shape of prediction input is {X_new.shape}")
Y_proba= model.predict(X_new)
Y_pred = model.predict_classes(X_new)

#predicted values
np.array(class_names)[Y_pred]


# In[ ]:





# In[ ]:





# In[30]:


np.array(class_names)[Y_pred]


# In[33]:


# Seeing images from rows and its predicted value
plt.rcParams['figure.figsize'] = (15, 5)
plt.suptitle("Predicted values",size=20)
plt.subplot(1,5,1)
plt.imshow(X_test[0])
plt.title(np.array(class_names)[Y_pred][0])


plt.subplot(1,5,2)
plt.imshow(X_test[1])
plt.title(np.array(class_names)[Y_pred][1])

plt.subplot(1,5,3)
plt.imshow(X_test[2])
plt.title(np.array(class_names)[Y_pred][2])

plt.subplot(1,5,4)
plt.imshow(X_test[3])
plt.title(np.array(class_names)[Y_pred][3])

plt.subplot(1,5,5)
plt.imshow(X_test[4])
plt.title(np.array(class_names)[Y_pred][4])


# # Saving the model for future use

# In[27]:


# remove the hash from below line to save model
# we have already saved the model
model.save("mnist_p.h5")


# # Model Training Done
# Now to make a GUI for live prediction we have two ways one to use tkinter module and other is use pygame.
# 

# In[ ]:




