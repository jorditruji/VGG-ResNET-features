from Data_utils import dataset, data_load
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Conv2DTranspose, Conv2D, Input, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16


# MINI VGG AUTOENCODER CUSTOM NET:
def mini_vgg(input_shape):
	#input images shape
	inputs = Input(input_shape)
	#CNN Coder
	out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
	out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
	out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
	#CNN DECODER:
	out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
	out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
	out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
	#create net
	depthnet = Model(inputs, out)
	depthnet.summary()



#CREATE VGG CONV NET (FULL CONVNET PRETRAINED WITH IMAGENET)
#vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(640, 480, 3))
inputs = Input(shape=(640, 480, 3))
#out = Dense(1, name='my_dense',activation='relu')(vgg.layers[-1].output)
#CONVOLUTIONAL BLOCK:
out= Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(out)
#out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
#out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
#out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)

'''
#DECONVOLUTIONAL BLOCK FOR FULL VGG16 NET :
#out =Conv2DTranspose(512,(3, 3),strides=(2, 2), padding='same', name='block5_deconv', activation='relu')(vgg.output)
#out =Conv2DTranspose(256,(3, 3),strides=(2, 2), padding='same', name='block4_deconv', activation='relu')(out)
#out =Conv2DTranspose(128,(3, 3),strides=(2, 2), padding='same', name='block3_deconv', activation='relu')(out)
out =Conv2DTranspose(64,(3, 3),strides=(2, 2), padding='same', name='block2_deconv', activation='relu')(out)
out =Conv2DTranspose(1,(3, 3),strides=(2, 2), padding='same', name='block1_deconv')(out)
'''
#FC+REGRESSION
out = Dropout(0.5)(out)
out=Conv2DTranspose(1,(3, 3),strides=(2, 2), padding='same', name='block5_deconv', activation='relu')(out)

#inp = vgg.input
depthnet = Model(inputs, out)
depthnet.summary()
optimizer=RMSprop(lr=0.01)


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



