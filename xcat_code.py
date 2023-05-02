import numpy as np
import random
import matplotlib.pyplot as plt
from cProfile import label
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
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
#
ct_4D = np.load('inputs/ct_4D.npy')
dvf = np.load('inputs/dvf.npy')
a_dvf = dvf.mean(axis=0)
from sklearn.decomposition import PCA
pca_3 = PCA(n_components=3, random_state=2046)
flat = np.reshape(dvf, (9, 70*256*256*3))
pca_3.fit(flat)
eigen_vect = pca_3.components_
eigen_val = pca_3.explained_variance_
print('Eigen Vectors Shape', eigen_vect.shape)
print('Eigen Values Shape', eigen_val.shape)
#
#
print('All is going ok')
a = 3+5
b = 6*a^3
c = np.random.random((256,256))
print('old->', a, 'new->' ,b)
plt.imshow(c)
plt.savefig('random.png')