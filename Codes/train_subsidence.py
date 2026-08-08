# Required libraries

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import pandas as pd
import  pyproj
import rasterio as rio
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import tensorflow as tf
from tensorflow import keras as ks
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from keras.regularizers import l2 , l1
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Dropout, AveragePooling2D, GlobalMaxPooling2D
import livelossplot
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import cv2

# x_train2=np.load('x_train2.npy')
# x_test=np.load('x_test.npy')
# x_val=np.load('x_val.npy')
# y_train2=np.load('y_train2.npy')
# y_test=np.load('y_test.npy')
# y_val=np.load('y_val.npy')

# ind_train2=np.load('ind_train2.npy')
# ind_test=np.load('ind_test.npy')
# ind_val=np.load('ind_val.npy')

#1401.04.19
#define the convolutional neural network architecture

model = Sequential()
model.add(Conv2D(32, (1,1), padding="same" ,input_shape=(30,30,9), activation="relu"))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (1,1), padding="same" ,input_shape=(30,30,6), activation="relu"))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(1024, (1,1), padding="same" ,input_shape=(30,30,6), activation="relu"))

model.add(Conv2D(1024, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(1024, activation="relu")) # 1024
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Dense(1024, activation="relu")) # 1024
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(512, activation="relu")) # 512
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(512, activation="relu")) # 512
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(256, activation="relu")) # 256
model.add(BatchNormalization())




model.add(Dense(1,activation='linear'))

model.summary()

plot_losses = livelossplot.PlotLossesKeras()

#model compile

model.compile(loss='mse' , optimizer= Adam(learning_rate=0.0001),metrics =['mae'])
checkpoint = ModelCheckpoint ('Bestmodel.h5',
                              save_best_only = True,
                              monitor = 'val_loss',
                              verbose =1)


# Train the model with the checkpoint callback
history = model.fit(x_train2, y_train2,
                    batch_size=128,
                    epochs=150,
                    callbacks= [plot_losses,checkpoint],
                    verbose=1,
                    validation_data=(x_val, y_val),
                    shuffle=True)

# Evaluate the model on training data
loss = model.evaluate(x_train2, y_train2)

# Extract the loss value (assuming it's the first element)
train_loss = loss[0]  # Get the first element for loss
print('Train Loss: {}'.format(round(float(train_loss), 2)))

# Evaluate on validation data
val_loss = model.evaluate(x_val, y_val)
val_loss_value = val_loss[0]  # Get the first element for validation loss
print('Validation Loss: {}'.format(round(float(val_loss_value), 2)))

# Evaluate on test data
test_loss = model.evaluate(x_test, y_test)
test_loss_value = test_loss[0]  # Get the first element for test loss
print('Test Loss: {}'.format(round(float(test_loss_value), 2)))

pred_train=model.predict(x_train2)
pred_test=model.predict(x_test)

utmx_train=np.load('utmx_train.npy')
utmy_train=np.load('utmy_train.npy')
y_train_real=np.load('y_train_real.npy')

utmx_test=np.load('utmx_test.npy')
utmy_test=np.load('utmx_test.npy')
y_test_real=np.load('y_test_real.npy')

scaler = MinMaxScaler()
scaler.fit(y_train_real)
pred_train_real = scaler.inverse_transform(pred_train)
mse = mean_squared_error(y_train_real, pred_train_real)
mse

pretrain=[]
for i in range(len(pred_train)):
  pretrain.append([utmx_train[i],utmy_train[i],pred_train[i][0]*(-10),y_train2[i]*(-10)])


pretest=[]
for i in range(len(pred_test)):
  pretest.append([utmx_test[i],utmy_test[i],pred_test[i][0]*(-10),y_test[i]*(-10)])

header = ['utmX','utmY','Prediction','Ground truth']
with open('pstrain.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(header)
    write.writerows(pretrain)

with open('pstest.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(header)
    write.writerows(pretest)

tf.keras.utils.plot_model(model, to_file='mymodel.pdf')