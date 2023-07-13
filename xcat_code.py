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
#
#########Inputs and Outputs##########
y= np.load('inputs/sample.npy',allow_pickle=True)
y = y[:,0]
v1 = y + abs(y.min())
y = np.round(v1/v1.max(), 2)
print('Output Size', y.shape)

mri_1= np.load('inputs/images_1.npy', allow_pickle=True)
mri_2 = np.load('inputs/images_2.npy',allow_pickle=True)
mri = np.concatenate((mri_1, mri_2))
mri = np.array([mri[i][:,125,:] for i in range(len(mri))])
print('MRI_size', mri.shape)

liver1 = np.load('inputs/im_liver_1.npy',allow_pickle=True)
liver2 = np.load('inputs/im_liver_2.npy',allow_pickle=True)
liver = np.concatenate((liver1,liver2))
liver = np.array([liver[i][:,125,:] for i in range(len(liver))])
print('LIVER_size', liver.shape)
ptv1 = np.load('inputs/im_ptv_1.npy',allow_pickle=True)
ptv2 = np.load('inputs/im_ptv_2.npy',allow_pickle=True)
ptv = np.concatenate((ptv1,ptv2))
ptv = np.array([ptv[i][:,125,:] for i in range(len(ptv))])
print('PTV_size', ptv.shape)

fig = plt.figure(1)
plt.figure(figsize=(23,8))
plt.subplot(2,5,1)
i= random.randrange(0,999)
print(i)
plt.imshow(mri[i], cmap='gray', aspect= 'auto') 
plt.contour(liver[i])
plt.contour(ptv[i], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,2)
i= random.randrange(100,199)
print(i)
plt.imshow(mri[i], cmap='gray', aspect= 'auto') 
plt.contour(liver[i])
plt.contour(ptv[i], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,3)
i= random.randrange(100,199)
print(i)
plt.imshow(mri[i], cmap='gray', aspect= 'auto') 
plt.contour(liver[i])
plt.contour(ptv[i], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,4)
i= random.randrange(0,99)
print(i)
plt.imshow(mri[i], cmap='gray', aspect= 'auto') 
plt.contour(liver[i])
plt.contour(ptv[i], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.savefig('Sanity_check.png', bbox_inches='tight')
#
#
# Model for MRI
i = Input(shape=(80, 256, 1))
x = Conv2D(filters=164, kernel_size=(8,8), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(8,8))(x)
x = Conv2D(filters=64, kernel_size=(6,6), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(6,6))(x)
x = Flatten()(x)
x = Dense(832, activation='relu')(x)
x = Dense(190, activation='relu')(x)
x = Dense(85, activation='relu')(x)
x = Dense(20, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model1 = Model(i, x)
#model.summary()
#adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model1.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
#
# Model for LIVER
i = Input(shape=(80, 256, 1))
x = Conv2D(filters=164, kernel_size=(8,8), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(8,8))(x)
x = Conv2D(filters=34, kernel_size=(8,8), activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(8,8))(x)
x = Flatten()(x)
x = Dense(832, activation='relu')(x)
x = Dense(410, activation='relu')(x)
x = Dense(200, activation='relu')(x)
x = Dense(105, activation='relu')(x)
x = Dense(10, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model2 = Model(i, x)
#model.summary()
#adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model2.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
#
# Model for PTV
i = Input(shape=(80, 256, 1))
x = Conv2D(filters=164, kernel_size=(8,8), activation='relu', padding='same')(i)
x = MaxPool2D(pool_size=(8,8))(x)
x = Flatten()(x)
x = Dense(232, activation='relu')(x)
x = Dense(110, activation='relu')(x)
x = Dense(45, activation='relu')(x)
x = Dense(10, activation='relu')(x)
x = Dense(1, activation='linear')(x)
model3 = Model(i, x)
#model.summary()
#adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model3.compile(loss='mean_squared_error', optimizer= 'adam', metrics=['mean_absolute_error'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
#
#
models= [model1, model1, model1, model1, model1]
x = ptv
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #random_state=1
X_train = X_train.reshape(160, 80, 256, 1)
X_test = X_test.reshape(40, 80, 256, 1)
predictions=[]
mean_abs_err=[]
root_mean_sq=[]
r=[]
fig2, ax = plt.subplots()
for model in models:   
    model3.fit(x=X_train, y= y_train, validation_data= (X_test, y_test), epochs=100, callbacks=[early_stop], verbose=0, batch_size=10)
    pred = model3.predict(X_test)
    s = r2_score(y_test, pred)
    b = mae(y_test, pred)
    #print('MAE-->>', mae(y_test, pred))
    a = sqrt(metrics.mean_squared_error(y_test, pred.ravel()))
    #print('MSE-->>', a)
    predictions.append(pred.ravel())
    mean_abs_err.append(b)
    root_mean_sq.append(a)
    r.append(s)
me = np.array(mean_abs_err)
rs = np.array(root_mean_sq)
predy = np.array(predictions)
r2 = np.array(r)
k1 = predy.mean(axis=0)
k2 = predy.std(axis=0)
new_dfw = pd.DataFrame({'Predicted Values':k1, 'True Values':y_test, 'Standard Deviation':k2})
plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k', alpha=.15)
plt.plot([0.05, 1], [0, 0.95], 'r--', linewidth=0.8, alpha=.15)
plt.plot([0, 0.95], [0.05, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
plt.annotate("$r^2$ = {:.2f}".format(r2.mean())+'$\pm$= {:.2f}'.format(r2.std()), (0, 0.9))
plt.plot([0.10, 1], [0, 0.90], 'r--', linewidth=0.8, alpha=.3)
plt.plot([0, 0.90], [0.10, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
sns.regplot(data=new_dfw, y='Predicted Values', x='True Values', scatter_kws=dict(color='k', s=20, alpha=1, marker='*'), line_kws=dict(color='orange', alpha=0.9))
ax.errorbar(data=new_dfw, y='Predicted Values', x='True Values', yerr=new_dfw['Standard Deviation'], fmt='none', capsize=0,  color='gray')
plt.xlim(-0.1,1.1)
plt.savefig('PTV_model_performance.png', bbox_inches='tight')
