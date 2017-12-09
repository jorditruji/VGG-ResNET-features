from Data_utils import * 
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Conv2DTranspose
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16

#CREATE VGG CONV NET
vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(640, 480, 3))

#out = Dense(1, name='my_dense',activation='relu')(vgg.layers[-1].output)
#DECONVOLUTIONAL BLOCK
out =Conv2DTranspose(512,(3, 3),strides=(2, 2), padding='same', name='block5_deconv', activation='relu')(vgg.output)
out =Conv2DTranspose(256,(3, 3),strides=(2, 2), padding='same', name='block4_deconv', activation='relu')(out)
out =Conv2DTranspose(128,(3, 3),strides=(2, 2), padding='same', name='block3_deconv', activation='relu')(out)
out =Conv2DTranspose(64,(3, 3),strides=(2, 2), padding='same', name='block2_deconv', activation='relu')(out)
out =Conv2DTranspose(1,(3, 3),strides=(2, 2), padding='same', name='block1_deconv')(out)

inp = vgg.input
depthnet = Model(inp, out)
depthnet.summary()

depthnet.compile(loss='mean_absolute_error', optimizer='rmsprop')

dataset = dataset(batch_size=6, samples_train=1000, samples_val=200)

history= model2.fit_generator(
	dataset.train_generator('/imatge/jmorera/work/train.txt'),
	nb_epoch = 20,
	verbose=1,
	validation_data=dataset.val_generator('/imatge/jmorera/work/val.txt'))



