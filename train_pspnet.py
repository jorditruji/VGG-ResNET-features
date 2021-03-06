from Data_utils import dataset, data_load
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Dense, Conv2DTranspose, Conv2D, Input, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Model
import numpy as np
from Models import vgg16
from layers_builder import *

input_shape=(640,480)

pspnet=build_pspnet( 1,50, input_shape)

pspnet.summary()


# initialize dataset
dataset = dataset.dataset(batch_size=6, samples_train=4000, samples_val=1500,normalize_depth=False)

history= pspnet.fit_generator(
	dataset.train_generator('/imatge/jmorera/work/train.txt'),
	nb_epoch = 50,
	verbose=1,
	steps_per_epoch=4500,
	validation_steps=1500,
	validation_data=dataset.val_generator('/imatge/jmorera/work/val.txt'))