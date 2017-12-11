from Data_utils import dataset, data_load
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Conv2DTranspose, Conv2D, Input, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16


# MINI VGG AUTOENCODER:
def mini_vgg(input_shape,extra_conv,decoder):
	#input images shape
	inputs = Input(input_shape)
	#CNN Coder
	out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
	out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
	if extra_conv:
		out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
		out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
	#CNN DECODER:
	if decoder:
		out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
		out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
		out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
		out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
		if extra_conv:
			out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
			out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
	#create net
	depthnet = Model(inputs, out)
	depthnet.summary()

#def pretrained_vgg(input_shape,extra_conv,decoder):

#CREATE VGG CONV NET (FULL CONVNET PRETRAINED WITH IMAGENET)
vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(640, 480, 3))
for i in range(12):
	vgg.layers.pop()

inputs = Input(shape=(640, 480, 3))
#out = Dense(1, name='my_dense',activation='relu')(vgg.layers[-1].output)
'''
#CONVOLUTIONAL BLOCK:
out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
#out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
#out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
#out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
'''
#drop+deconv
#out = Dropout(0.5)(out)
out=Conv2DTranspose(128,(3, 3),strides=(2, 2), padding='same', name='block4_deconv', activation='relu')(vgg.layers[-1].output)
out=Dropout(0.5)(out)
out=Conv2DTranspose(1,(3, 3),strides=(2, 2), padding='same', name='block5_deconv', activation='relu')(out)

#inp = vgg.input
depthnet = Model(vgg.input, out)
depthnet.summary()



depthnet.compile(loss='mean_absolute_error', optimizer='adam')
# initialize dataset
dataset = dataset.dataset(batch_size=16, samples_train=1000, samples_val=300,normalize_depth=False)

history= depthnet.fit_generator(
	dataset.train_generator('/imatge/jmorera/work/train.txt'),
	nb_epoch = 50,
	verbose=1,
	steps_per_epoch=20,
	validation_steps=6,
	validation_data=dataset.val_generator('/imatge/jmorera/work/val.txt'))



