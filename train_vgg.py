from Data_utils import dataset, data_load
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Conv2DTranspose, Conv2D, Input, MaxPooling2D
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16

#CREATE VGG CONV NET
#vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(640, 480, 3))
inputs = Input(shape=(640, 480, 3))
#out = Dense(1, name='my_dense',activation='relu')(vgg.layers[-1].output)
#DECONVOLUTIONAL BLOCK
out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

#out =Conv2DTranspose(512,(3, 3),strides=(2, 2), padding='same', name='block5_deconv', activation='relu')(vgg.output)
#out =Conv2DTranspose(256,(3, 3),strides=(2, 2), padding='same', name='block4_deconv', activation='relu')(out)
out =Conv2DTranspose(128,(3, 3),strides=(2, 2), padding='same', name='block3_deconv', activation='relu')(out)
out =Conv2DTranspose(64,(3, 3),strides=(2, 2), padding='same', name='block2_deconv', activation='relu')(out)
out =Conv2DTranspose(1,(3, 3),strides=(2, 2), padding='same', name='block1_deconv')(out)

inp = vgg.input
depthnet = Model(inputs, out)
depthnet.summary()

depthnet.compile(loss='mean_absolute_error', optimizer='sgd')

dataset = dataset.dataset(batch_size=6, samples_train=7000, samples_val=3000)

history= depthnet.fit_generator(
	dataset.train_generator('/imatge/jmorera/work/train.txt'),
	nb_epoch = 50,
	verbose=1,
	steps_per_epoch=1000,
	validation_steps=350,
	validation_data=dataset.val_generator('/imatge/jmorera/work/val.txt'))



