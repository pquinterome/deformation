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
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import shapiro
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import RocCurveDisplay, auc, precision_score, recall_score, f1_score, roc_curve
from numpy import interp
from sklearn.decomposition import PCA
#
ct_4D = np.load('inputs/ct_4D.npy', allow_pickle=True)
dvf = np.load('inputs/dvf.npy', allow_pickle=True)
print('DVF shape', dvf.shape)
a_dvf = dvf.mean(axis=0)
###### PCA #################
pca_3 = PCA(n_components=3, random_state=2046)
flat = np.reshape(dvf, (9, 70*256*256*3))
pca_3.fit(flat)
eigen_vect = pca_3.components_
eigen_val = pca_3.explained_variance_
print('Eigen Vectors Shape', eigen_vect.shape)
print('Eigen Values Shape', eigen_val.shape)
##############################
sampl = np.load('inputs/sample.npy', allow_pickle=True)
vect = eigen_vect.reshape(3, 70, 256, 256, 3)
d = []
l = []
for r in range(500):   
    val = sampl[r]/10000
    l.append(val)
    vectors = a_dvf + sum(np.array([vect[i]*val[i] for i in range(len(sampl[8]))]))
    displacement_image = sitk.GetImageFromArray(vectors, isVector=True)
    # Set the displacement image's origin and spacing to relevant values
    displacement_image.SetOrigin((0,0,0))
    displacement_image.SetSpacing((1,1,1))
    tx = sitk.DisplacementFieldTransform(displacement_image)
    moving_image = sitk.GetImageFromArray(np.ascontiguousarray(ct_4D[0][:,:,:]))
    #sitk.WriteImage(fixed_image, os.path.join(OUTPUT_DIR, f"mr/fixed_image{i}.mha"))
    syn_deform = sitk.Resample(moving_image, tx)
    syn_deform = sitk.GetArrayViewFromImage(syn_deform) 
    syn_deform = syn_deform[:,:,:]   
    d.append(syn_deform)
ll = np.array(l)
dd = np.array(d)
#
np.save('eigen_val_1.npy', ll)
np.save('dataset_imgs_1.npy', dd)
print('eigenvalues shape', ll.shape)
print('dataset_images', dd.shape)
print('All is going ok')
#
plt.figure(figsize=(35,18))
plt.subplot(3,5,1)
plt.imshow(ct_4D[0][:,129,:], cmap='gray')
plt.axhline(y=35, color='r', linestyle='--', lw=0.5)
plt.axhline(y=64, color='c', linestyle='-', lw=1)
#plt.ylim(0, 70)
plt.subplot(3,5,2)
plt.imshow(dd[0][:,129,:], cmap='gray')
plt.axhline(y=35, color='r', linestyle='--', lw=0.5)
plt.axhline(y=64, color='c', linestyle='-', lw=1)
#
plt.subplot(3,5,3)
plt.imshow(dd[1][:,129,:], cmap='gray')
plt.axhline(y=35, color='r', linestyle='--', lw=0.5)
plt.axhline(y=64, color='c', linestyle='-', lw=1)
#
plt.subplot(3,5,4)
plt.imshow(dd[2][:,129,:], cmap='gray')
plt.axhline(y=35, color='r', linestyle='--', lw=0.5)
plt.axhline(y=64, color='c', linestyle='-', lw=1)

plt.savefig('random.png')