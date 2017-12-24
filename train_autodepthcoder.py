from Data_utils import dataset, data_load
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Conv2DTranspose, Conv2D, Input, MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16



def build_model(encoder_dim=50, bottleneck_dim=20):
	inputs = Input(shape=(640, 480, 3))
	out= Conv2D(64, (9, 9), activation='relu', padding='same', name='block1_conv1')(inputs)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(out)
	out = Conv2D(128, (5, 5), activation='relu', padding='same', name='block2_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(out)
	out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(out)
	out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(out)
	#out= Dropout(0.25)(out)
	out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_deconv1')(out)
	out=UpSampling2D((2,2), name='up1')(out)
	out = Conv2D(128, (5, 5), activation='relu', padding='same', name='block2_deconv1')(out)
	out=UpSampling2D((2,2), name='up2')(out)
	out= Conv2D(64, (9, 9), activation='relu', padding='same', name='block1_deconv1')(out)
	out=UpSampling2D((2,2), name='up3')(out)
	'''
	out =Flatten()(out)
	out = Dense(256, name='my_dense',activation='relu')(out)
	out=Reshape((256, 80, 60))(out)
	'''
	depthnet = Model(inputs, out)
	depthnet.summary()
	return depthnet
net = build_model()
net.summary()




depthnet.compile(loss='mean_absolute_error', optimizer='adam')
# initialize dataset
dataset = dataset.dataset(batch_size=12, samples_train=600, samples_val=150,normalize_depth=False)

history= depthnet.fit_generator(
	dataset.train_generator('/imatge/jmorera/work/train.txt'),
	nb_epoch = 50,
	verbose=1,
	steps_per_epoch=50,
	validation_steps=10,
	validation_data=dataset.val_generator('/imatge/jmorera/work/val.txt'))
