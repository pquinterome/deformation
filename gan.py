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
######################################
#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus: 
#    tf.config.experimental.set_memory_growth(gpu, True)

# Bringing in matplotlib for viz stuff
from matplotlib import pyplot as plt
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D

#########Inputs and Outputs##########
#ct = np.load('inputs/ct2.npy', allow_pickle=True)
cbct = np.load('inputs/cbct.npy', allow_pickle=True)
#ct = ct.reshape(3920, 512, 512,1)
cbct = cbct.reshape(3920,128,128,1)
X = cbct[:200,:,:,:]
#y = cbct
ds = tf.data.Dataset.from_tensor_slices(X)
ds = ds.cache()
ds = ds.batch(10)
ds = ds.prefetch(64)
print('ds.as_numpy_iterator()', ds.as_numpy_iterator().next().shape)
print('ds.as_numpy_iterator().next()[0]', ds.as_numpy_iterator().next()[0].shape)
#
#
#
dataiterator = ds.as_numpy_iterator()
fig = plt.figure(1)
fig, ax = plt.subplots(ncols=4, figsize=(10,5))
# Loop four times and get images 
for idx in range(4): 
    # Grab an image and label
    sample = dataiterator.next()[idx]
    # Plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(sample))
    # Appending the image label as the plot title 
    ax[idx].title.set_text(idx)
plt.savefig('outputs/Sanity_iterator.png', bbox_inches='tight')
#
#
#
#
def build_generator(): 
    model = Sequential()
    
    # Takes in random values and reshapes it to 7x7x128
    # Beginnings of a generated image
    model.add(Dense(8*8*100, input_dim=10))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((8,8,100)))
    
    # Upsampling block 1 
    model.add(UpSampling2D())
    model.add(Conv2D(100, 5, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Upsampling block 2 
    model.add(UpSampling2D())
    model.add(Conv2D(100, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 3 
    model.add(UpSampling2D())
    model.add(Conv2D(100, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    # Upsampling block 4 
    model.add(UpSampling2D())
    model.add(Conv2D(100, 5, padding='same'))
    model.add(LeakyReLU(0.2))

    ## Upsampling block 5 
    #model.add(UpSampling2D())
    #odel.add(Conv2D(100, 5, padding='same'))
    #model.add(LeakyReLU(0.2))

    ## Upsampling block 6 
    #model.add(UpSampling2D())
    #model.add(Conv2D(100, 5, padding='same'))
    #model.add(LeakyReLU(0.2))
    
    # Convolutional block 1
    model.add(Conv2D(100, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Convolutional block 2
    model.add(Conv2D(100, 4, padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Conv layer to get to one channel
    model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))
    
    return model
generator = build_generator()
generator.summary()
#
# Generate new fashion
img = generator.predict(np.random.randn(4,10,1))
# Setup the subplot formatting 
fig = plt.figure(2)
fig, ax = plt.subplots(ncols=4, figsize=(10,15))
# Loop four times and get images 
for idx, img in enumerate(img): 
    # Plot the image using a specific subplot 
    ax[idx].imshow(np.squeeze(img))
    # Appending the image label as the plot title 
    ax[idx].title.set_text(idx)
plt.savefig('outputs/Sanity_generator.png', bbox_inches='tight')
#
#
#
def build_discriminator(): 
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(32, 5, input_shape = (128,128,1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Second Conv Block
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Third Conv Block
    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    # Fourth Conv Block
    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    # Fifth Conv Block
    model.add(Conv2D(512, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    
    ## Fourth Conv Block
    #model.add(Conv2D(512, 50))
    #model.add(LeakyReLU(0.2))
    #model.add(Dropout(0.4))

    
    # Flatten then pass to dense layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    return model 
discriminator = build_discriminator()
discriminator.summary()
img = generator.predict(np.random.randn(4,10,1))
discriminator.predict(img)
#
# Adam is going to be the optimizer for both
from tensorflow.keras.optimizers.legacy import Adam
# Binary cross entropy is going to be the loss for both 
from tensorflow.keras.losses import BinaryCrossentropy
#
g_opt = Adam(learning_rate=0.0001) 
d_opt = Adam(learning_rate=0.00001) 
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()
# Importing the base model class to subclass our training step 
from tensorflow.keras.models import Model
#
#
#
class FashionGAN(Model): 
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class 
        super().__init__(*args, **kwargs)
        
        # Create attributes for gen and disc
        self.generator = generator 
        self.discriminator = discriminator 
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs): 
        # Compile with base class
        super().compile(*args, **kwargs)
        
        # Create attributes for losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 

    def train_step(self, batch):
        # Get the data 
        real_images = batch
        fake_images = self.generator(tf.random.normal((100, 10, 1)), training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_images, training=True) 
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            
            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)
            
            # Add some noise to the TRUE outputs
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply backpropagation - nn learn 
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as g_tape: 
            # Generate some new images
            gen_images = self.generator(tf.random.normal((100,10,1)), training=True)
                                        
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_images, training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
            
        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        return {"d_loss":total_d_loss, "g_loss":total_g_loss}
# Create instance of subclassed model
fashgan = FashionGAN(generator, discriminator)
# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)
import os
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=10):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('images', f'generated_img_{epoch}_{i}.png'))
#
#
# Recommend 2000 epochs
print('ready to train')
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
with strategy.scope():
    hist = fashgan.fit(ds, epochs=200, callbacks=[ModelMonitor()])
fig = plt.figure(2)
plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.savefig('outputs/Training_gan.png', bbox_inches='tight')
generator.save('outputs/generator.h5')
discriminator.save('outputs/discriminator.h5')