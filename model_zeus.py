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
# How are you
#########Inputs and Outputs##########



y= np.load('inputs/zeus/sample.npy',allow_pickle=True)
y = y[:,0]
y = y[:400]
y = np.array([-y[i]/y.min() if y[i]<0 else y[i]/y.max() for i in range(len(y))])
print('Output Size', y.shape)

mri_1= np.load('inputs/zeus/im_ct_1.npy', allow_pickle=True)
x = mri_1[:400]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
X_train = X_train.reshape(320, x.shape[1], x.shape[2], x.shape[3], 1)
X_test = X_test.reshape(80,x.shape[1], x.shape[2], x.shape[3], 1)

# Model for MRI
i = Input(shape=(x.shape[1], x.shape[2], x.shape[3], 1))
x = Conv3D(filters=64, kernel_size=(8,8,8), activation='relu', padding='same')(i)
x = MaxPool3D(pool_size=(8,8, 8))(x)
#x = BatchNormalization()(x)
x = Conv3D(filters=34, kernel_size=(4,4, 4), activation='relu', padding='same')(x)
x = MaxPool3D(pool_size=(4,4,4))(x)
#x = BatchNormalization()(x)
#x = Dropout(0.5)(x)
x = Flatten()(x)
#x = Dense(832, activation='relu')(x)
#x = Dropout(0.1)(x)
x = Dense(410, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dense(105, activation='relu')(x)
x = Dense(10, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model1 = Model(i, x)
#model1.summary()
#adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model1.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history= model1.fit(x=X_train, y= y_train, validation_data= (X_test, y_test), epochs=100, callbacks=[early_stop], verbose=2, batch_size=10)


model1 = tf.keras.models.load_model('outputs/model_2_reg.h5', compile=False)
print('all ok (:')
model1.save('outputs/z_model.h5')

pred = model1.predict(X_test)

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
plt.scatter(x=y_test, y=pred, edgecolors='k', color='r', alpha=0.7, label='Q1')
#plt.scatter(x=y_test[:,1], y=pred[:,1], edgecolors='k', color='g', alpha=0.7, label='Q2')
#plt.scatter(x=y_test[:,2], y=pred[:,2], edgecolors='k', color='cyan', alpha=0.7, label='Q3')
plt.plot([1, -1], [1, -1], linestyle='--', lw=1, color='k', alpha=.15)
plt.plot([-0.95, 1], [-1, 0.95], 'r--', linewidth=0.8, alpha=.15)
plt.plot([-1, 0.95], [-0.95, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
plt.plot([-0.90, 1], [-1, 0.90], 'r--', linewidth=0.8, alpha=.3)
plt.plot([-1, 0.90], [-0.90, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig('Zeus_model_performance.png', bbox_inches='tight')


print('MRI_MODEL')

np.save('outputs/z_y_test.npy', y_test)
np.save('outputs/z_predy.npy', pred)
me = mae(y_test, pred)
print(me)
np.save('outputs/z_mae.npy', me)
rs = sqrt(metrics.mean_squared_error(y_test, pred.ravel()))
print(rs)
np.save('outputs/z_rmse.npy', rs)
print('***MRI_Model: DONE***')