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
y= np.load('inputs/sample.npy',allow_pickle=True)
y = y[:,0]
y1 = y[:400]
y2 = y[-400:]
y = np.concatenate((y1,y2))
y = np.array([-y[i]/y.min() if y[i]<0 else y[i]/y.max() for i in range(len(y))])
print('Output Size', y.shape)

mri_1= np.load('inputs/im_ct_1.npy', allow_pickle=True)
mri_1 = mri_1[:400]
mri_2 = np.load('inputs/im_ct_2.npy',allow_pickle=True)
mri_2 = mri_2[-400:]
mri = np.concatenate((mri_1, mri_2))    #.astype('int32')
#mri = np.array([mri[i][:,125,:] for i in range(len(mri))])
print('MRI_size', mri.shape)

#liver1 = np.load('inputs/im_liver_1.npy',allow_pickle=True)
#liver1 = liver1[:500]
#liver2 = np.load('inputs/im_liver_2.npy',allow_pickle=True)
#liver2 = liver2[-500:]
#liver = np.concatenate((liver1,liver2))
##liver = np.array([liver[i][:,125,:] for i in range(len(liver))])
#print('LIVER_size', liver.shape)
#ptv1 = np.load('inputs/im_ptv_1.npy',allow_pickle=True)
#ptv1 = ptv1[:500]
#ptv2 = np.load('inputs/im_ptv_2.npy',allow_pickle=True)
#ptv2 = ptv2[-500:]
#ptv = np.concatenate((ptv1,ptv2))
##ptv = np.array([ptv[i][:,125,:] for i in range(len(ptv))])
#print('PTV_size', ptv.shape)

fig = plt.figure(1)
plt.figure(figsize=(23,8))
plt.subplot(2,5,1)
i= random.randrange(0,50)
print(i)
plt.imshow(mri[i][:,125,:], cmap='gray', aspect= 'auto') 
#plt.contour(liver[i][:,125,:])
#plt.contour(ptv[i][:,125,:], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,2)
i= random.randrange(50,199)
print(i)
plt.imshow(mri[i][:,125,:], cmap='gray', aspect= 'auto') 
#plt.contour(liver[i][:,125,:])
#plt.contour(ptv[i][:,125,:], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,3)
i= random.randrange(10,150)
print(i)
plt.imshow(mri[i][:,125,:], cmap='gray', aspect= 'auto') 
#plt.contour(liver[i][:,125,:])
#plt.contour(ptv[i][:,125,:], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.subplot(2,5,4)
i= random.randrange(15,199)
print(i)
plt.imshow(mri[i][:,125,:], cmap='gray', aspect= 'auto') 
#plt.contour(liver[i][:,125,:])
#plt.contour(ptv[i][:,125,:], cmap='Pastel1')
plt.xlabel(y[i])
plt.axhline(y=75, color='c', linestyle='-', lw=1)
plt.axhline(y=65, color='c', linestyle=':', lw=1)
plt.axhline(y=50, color='r', linestyle='-', lw=1)
plt.axhline(y=40, color='r', linestyle=':', lw=1)
plt.ylim(0,80)
plt.title(i)
plt.savefig('outputs/Sanity_check.png', bbox_inches='tight')
#

xw = mri
X_train, X_test, y_train, y_test = train_test_split(xw, y, test_size=0.2) #random_state=1
print(X_train.shape)
print(X_test.shape)
X_train = X_train.reshape(640, 80, 256, 256, 1)
X_test = X_test.reshape(160, 80, 256, 256, 1)
print(X_train.shape)
print(X_test.shape)



model1 = tf.keras.models.load_model('outputs/model_2_reg.h5', compile=False)
print('all ok (:')

predictions=[]
mean_abs_err=[]
root_mean_sq=[]
yey=[]
r=[]
models= [model1, model1, model1, model1, model1]
early_stop = EarlyStopping(monitor='val_loss', patience=5)
fig3 = plt.subplots()
for model in models:
    xw = mri
    X_train, X_test, y_train, y_test = train_test_split(xw, y, test_size=0.2) #random_state=1
    print(X_train.shape)
    print(X_test.shape)
    X_train = X_train.reshape(640, 80, 256, 256, 1)
    X_test = X_test.reshape(160, 80, 256, 256, 1)     
    pred = model1.predict(X_test)
    s = r2_score(y_test, pred)
    b = mae(y_test, pred)
    #print('MAE-->>', mae(y_test, pred))
    a = sqrt(metrics.mean_squared_error(y_test, pred.ravel()))
    #print('MSE-->>', a)
    predictions.append(pred.ravel())
    mean_abs_err.append(b)
    root_mean_sq.append(a)
    yey.append(y_test)
    r.append(s)
wey = np.array(yey)
me = np.array(mean_abs_err)
rs = np.array(root_mean_sq)
predy = np.array(predictions)
r2 = np.array(r)
#k1 = predy.mean(axis=0)
#k2 = predy.std(axis=0)
#new_dfw = pd.DataFrame({'Predicted Values':k1, 'True Values':y_test, 'Standard Deviation':k2})
plt.plot([1, -1], [1, -1], linestyle='--', lw=1, color='k', alpha=.15)
plt.plot([-0.95, 1], [-1, 0.95], 'r--', linewidth=0.8, alpha=.15)
plt.plot([-1, 0.95], [-0.95, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
plt.annotate("$r^2$ = {:.2f}".format(r2.mean())+'$\pm$ {:.2f}'.format(r2.std()), (-1, 0.9))
plt.plot([-0.90, 1], [-1, 0.90], 'r--', linewidth=0.8, alpha=.3)
plt.plot([-1, 0.90], [-0.90, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
plt.plot(predy[0], wey[0], 'ko', alpha=0.35)
plt.plot(predy[1], wey[1], 'ko', alpha=0.35)
plt.plot(predy[2], wey[2], 'ko', alpha=0.35)
plt.plot(predy[3], wey[3], 'ko', alpha=0.35)
plt.plot(predy[4], wey[4], 'ko', alpha=0.35)
#sns.regplot(data=new_dfw, y='Predicted Values', x='True Values', scatter_kws=dict(color='k', s=20, alpha=1, marker='*'), line_kws=dict(color='orange', alpha=0.9))
#ax3.errorbar(data=new_dfw, y='Predicted Values', x='True Values', yerr=new_dfw['Standard Deviation'], fmt='none', capsize=0,  color='gray')
plt.xlim(-1.1,1.1)
plt.savefig('outputs/MRI_model_performance_3d.png', bbox_inches='tight')
print('MRI_MODEL')

np.save('outputs/y_test.npy', wey)
np.save('outputs/predy.npy', predy)
np.save('outputs/mae.npy', me)
np.save('outputs/rmse.npy', rs)
print('***MRI_Model: DONE***')
#
#
#
#models= [model1, model1, model1, model1, model1]
#x = ptv
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #random_state=1
#X_train = X_train.reshape(800, 80, 256, 256, 1)
#X_test  = X_test.reshape(200, 80, 256, 256, 1)
#predictions=[]
#mean_abs_err=[]
#root_mean_sq=[]
#r=[]
#fig2, ax = plt.subplots()
#for model in models:   
#    model3.fit(x=X_train, y= y_train, validation_data= (X_test, y_test), epochs=100, callbacks=[early_stop], verbose=1, batch_size=10)
#    pred = model3.predict(X_test)
#    s = r2_score(y_test, pred)
#    b = mae(y_test, pred)
#    #print('MAE-->>', mae(y_test, pred))
#    a = sqrt(metrics.mean_squared_error(y_test, pred.ravel()))
#    #print('MSE-->>', a)
#    predictions.append(pred.ravel())
#    mean_abs_err.append(b)
#    root_mean_sq.append(a)
#    r.append(s)
#me = np.array(mean_abs_err)
#rs = np.array(root_mean_sq)
#predy = np.array(predictions)
#r2 = np.array(r)
#k1 = predy.mean(axis=0)
#k2 = predy.std(axis=0)
#new_dfw = pd.DataFrame({'Predicted Values':k1, 'True Values':y_test, 'Standard Deviation':k2})
#plt.plot([1, -1], [1, -1], linestyle='--', lw=1, color='k', alpha=.15)
#plt.plot([-0.95, 1], [-1, 0.95], 'r--', linewidth=0.8, alpha=.15)
#plt.plot([-1, 0.95], [-0.95, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
#plt.annotate("$r^2$ = {:.2f}".format(r2.mean())+'$\pm$= {:.2f}'.format(r2.std()), (-1, 0.9))
#plt.plot([-0.90, 1], [-1, 0.90], 'r--', linewidth=0.8, alpha=.3)
#plt.plot([-1, 0.90], [-0.90, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
#sns.regplot(data=new_dfw, y='Predicted Values', x='True Values', scatter_kws=dict(color='k', s=20, alpha=1, marker='*'), line_kws=dict(color='orange', alpha=0.9))
#ax.errorbar(data=new_dfw, y='Predicted Values', x='True Values', yerr=new_dfw['Standard Deviation'], fmt='none', capsize=0,  color='gray')
#plt.xlim(-1.1,1.1)
#plt.savefig('outputs/PTV_model_performance.png', bbox_inches='tight')
#print('PTV_MODEL')
#print('y_test', y_test)
#print('Predictions', predy)
#print('MAE', me)
#print('RMSE', rs)
#print('***PTV_Model: DONE***')
#
#
#
#
#models= [model1, model1, model1, model1, model1]
#x = liver
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #random_state=1
#X_train = X_train.reshape(800, 80, 256, 256, 1)
#X_test = X_test.reshape(200, 80, 256, 256, 1)
#predictions=[]
#mean_abs_err=[]
#root_mean_sq=[]
#r=[]
#fig3, ax2 = plt.subplots()
#for model in models:   
#    model2.fit(x=X_train, y= y_train, validation_data= (X_test, y_test), epochs=100, callbacks=[early_stop], verbose=0, batch_size=10)
#    pred = model2.predict(X_test)
#    s = r2_score(y_test, pred)
#    b = mae(y_test, pred)
#    #print('MAE-->>', mae(y_test, pred))
#    a = sqrt(metrics.mean_squared_error(y_test, pred.ravel()))
#    #print('MSE-->>', a)
#    predictions.append(pred.ravel())
#    mean_abs_err.append(b)
#    root_mean_sq.append(a)
#    r.append(s)
#me = np.array(mean_abs_err)
#rs = np.array(root_mean_sq)
#predy = np.array(predictions)
#r2 = np.array(r)
#k1 = predy.mean(axis=0)
#k2 = predy.std(axis=0)
#new_dfw = pd.DataFrame({'Predicted Values':k1, 'True Values':y_test, 'Standard Deviation':k2})
#plt.plot([1, -1], [1, -1], linestyle='--', lw=1, color='k', alpha=.15)
#plt.plot([-0.95, 1], [-1, 0.95], 'r--', linewidth=0.8, alpha=.15)
#plt.plot([-1, 0.95], [-0.95, 1], 'r--', linewidth=0.8, alpha=.15, label='$\pm$ 5%')
#plt.annotate("$r^2$ = {:.2f}".format(r2.mean())+'$\pm$= {:.2f}'.format(r2.std()), (-1, 0.9))
#plt.plot([-0.90, 1], [-1, 0.90], 'r--', linewidth=0.8, alpha=.3)
#plt.plot([-1, 0.90], [-0.90, 1], 'r--', linewidth=0.8, alpha=.3, label='$\pm$ 10%')
#sns.regplot(data=new_dfw, y='Predicted Values', x='True Values', scatter_kws=dict(color='k', s=20, alpha=1, marker='*'), line_kws=dict(color='orange', alpha=0.9))
#ax2.errorbar(data=new_dfw, y='Predicted Values', x='True Values', yerr=new_dfw['Standard Deviation'], fmt='none', capsize=0,  color='gray')
#plt.xlim(-1.1,1.1)
#plt.savefig('outputs/LIVER_model_performance.png', bbox_inches='tight')
#print('LIVER_MODEL')
#print('y_test', y_test)
#print('Predictions', predy)
#print('MAE', me)
#print('RMSE', rs)
#print('***LIVER_Model: DONE***')