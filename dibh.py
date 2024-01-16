import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.pyplot import figure
import numpy as np
import seaborn as sns
import scipy
import scipy.io
import scipy.ndimage
from scipy.spatial.distance import directed_hausdorff
import scipy.ndimage
import random
import gc
from skimage import metrics
from skimage import measure
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_error as mae
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, MaxPool2D, MaxPool3D, Flatten, Dropout, GlobalMaxPooling3D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import h5py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Hi
#########Inputs and Outputs##########
directory = './dibh_project/pca_deformed/' #----->>> Changing this location
path =  sorted(os.listdir(directory))
mri_dir = [directory+file for file in path if file.startswith("p")]
train = mri_dir[:80]
test = mri_dir[-20:]


y= np.load('./dibh_project/sample2.npy',allow_pickle=True)
y1 = y[:,:80]
y3 = np.array([y1[i]-y1[i].min() for i in range(len(y1))])
y = np.array([y3[i]/y3[i].max() for i in range(len(y3))])
y_train = y.reshape((80,3, 1))
print(y_train.shape)
y2 = y[:,-20:]
y3 = np.array([y2[i]-y2[i].min() for i in range(len(y2))])
y = np.array([y3[i]/y3[i].max() for i in range(len(y3))])
y_test = y.reshape((20,3, 1))
print(y_test.shape)

y_trainy = y_train
y_testy = y_test

######## DATASET GENERATOR ####################
AUTOTUNE = tf.data.AUTOTUNE
batch_size= 3
def preprocess_image_train(image):
  image = image
  return image

y_train = tf.data.Dataset.from_tensor_slices(y_trainy)
y_train = y_train.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(1)

y_test = tf.data.Dataset.from_tensor_slices(y_testy)
y_test = y_test.cache().map(preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(1)


def read_npy_file(item):
    data = np.load(item.numpy().decode())
    return data.astype(np.float64)

#file_list = ['/foo/bar.npy', '/foo/baz.npy']

dataset_train = tf.data.Dataset.from_tensor_slices(train)
dataset_train = dataset_train.cache().map(lambda item:(tf.py_function(read_npy_file, [item], [tf.float64,]))).batch(1)
dataset_test = tf.data.Dataset.from_tensor_slices(test)
dataset_test = dataset_test.cache().map(lambda item:(tf.py_function(read_npy_file, [item], [tf.float64,]))).batch(1)
#dataset = dataset.map(lambda item:tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))

train_dataset = tf.data.Dataset.zip((dataset_train, y_train))
test_dataset = tf.data.Dataset.zip((dataset_test, y_test))

print(train_dataset.element_spec)

########### MODELING ###################

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
with strategy.scope():
  i = Input(shape=(89, 576, 576, 1))
  x = Conv3D(filters=64, kernel_size=(8,8,8), activation='relu', padding='same')(i)
  x = MaxPool3D(pool_size=(8,8,8))(x)
  x = Conv3D(filters=32, kernel_size=(6,6,6), activation='relu', padding='same')(x)
  x = MaxPool3D(pool_size=(6,6,6))(x)
  x = Flatten()(x)
  x = Dense(780, activation='relu')(x)
  x = Dense(190, activation='relu')(x)
  #x = Dense(85, activation='relu')(x)
  #x = Dense(20, activation='relu')(x)
  x = Dense(3, activation='linear')(x)
  model1 = Model(i, x)
  model1.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])



#i = Input(shape=(89, 576, 576, 1))
#x = Conv3D(filters=64, kernel_size=(8,8,8), activation='relu', padding='same')(i)
#x = MaxPool3D(pool_size=(8,8,8))(x)
#x = Conv3D(filters=32, kernel_size=(6,6,6), activation='relu', padding='same')(x)
#x = MaxPool3D(pool_size=(6,6,6))(x)
#x = Flatten()(x)
#x = Dense(780, activation='relu')(x)
#x = Dense(190, activation='relu')(x)
##x = Dense(85, activation='relu')(x)
##x = Dense(20, activation='relu')(x)
#x = Dense(3, activation='linear')(x)
#model1 = Model(i, x)
#model1.summary()

#model1.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history= model1.fit(train_dataset, validation_data=test_dataset, epochs=5, callbacks=[early_stop], verbose=2)


pred = model1.predict(dataset_test)
fig = plt.figure(1)
plt.figure(figsize=(20,18))
plt.subplot(3,4,1)
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.subplot(3,4,2)
plt.title('Mean Absolute Error')
plt.plot(history.history['mean_absolute_error'], label='mae')
plt.plot(history.history['val_mean_absolute_error'], label='val_mae')
plt.legend()
plt.subplot(3,4,3)
plt.scatter(x=y_testy, y=pred, edgecolors='k', color='r', alpha=0.7, label='Q1')
plt.scatter(x=y_testy[:,1], y=pred[:,1], edgecolors='k', color='g', alpha=0.7, label='Q2')
plt.scatter(x=y_testy[:,2], y=pred[:,2], edgecolors='k', color='cyan', alpha=0.7, label='Q3')
plt.plot([1, -1], [1, -1], linestyle='--', lw=1, color='k', alpha=.15)
plt.plot([-0.95, 1], [-1, 0.95], 'r--', linewidth=0.8, alpha=.15)
plt.plot([-1, 0.95], [-0.95, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
plt.plot([-0.90, 1], [-1, 0.90], 'r--', linewidth=0.8, alpha=.3)
plt.plot([-1, 0.90], [-0.90, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend();
plt.savefig('./dibh_project/errors.png', bbox_inches='tight')
#
#
