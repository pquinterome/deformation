import numpy as np
import random
import matplotlib.pyplot as plt
from cProfile import label
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pyparsing import alphas
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPool3D, Flatten, Dropout, GlobalMaxPooling3D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import RocCurveDisplay, auc, precision_score, recall_score, f1_score, roc_curve
from numpy import interp
from sklearn.decomposition import PCA
#
#########Inputs and Outputs##########
pca_val= np.load('inputs/sample.npy',allow_pickle=True)/10000
pca = pca_val[:100]
print('labels size',pca.shape)
image_1= np.load('inputs/images_1.npy', allow_pickle=True)
image_1=image_1[:100]
print('Inputs size', image_1.shape)
X_train, X_test, y_train, y_test = train_test_split(image_1, pca, test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(80, 70, 256, 256,1)
X_test = X_test.reshape(20, 70, 256, 256,1)
batch_size=10
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(batch_size)
############Model##################
i = Input(shape=(70, 256, 256, 1))
x = Conv3D(filters=32, kernel_size=(6,6,6), activation='relu', padding='same')(i)
x = MaxPool3D(pool_size=(6,6,6))(x)
x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Conv3D(filters=32, kernel_size=(6,6,6), activation='relu', padding='same')(x)
x = MaxPool3D(pool_size=(6,6,6))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(180, activation='relu')(x)
x = Dense(90, activation='relu')(x)
x = Dense(3, activation='linear')(x)
model = Model(i, x)
model.compile(loss='mean_squared_error', optimizer= "adam", metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')
############Model Fit###############
history = model.fit(train_dataset, validation_data= val_dataset, epochs=100, callbacks=[early_stop], verbose=2)
pred = model.predict(X_test)
print('Total mae', mae(y_test, pred))
##### Loss during training##########
plt.figure(figsize=(35,18))
plt.subplot(1,4,1)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#
plt.subplot(1,4,2)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.legend()
#
plt.subplot(1,4,3)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
#
plt.subplot(1,4,4)
plt.title('Mean Absolute Error')
plt.plot(history.history['mean_absolute_error'], label='mae')
plt.plot(history.history['val_mean_absolute_error'], label='val_mae')
plt.legend()
#
plt.savefig('random.png')