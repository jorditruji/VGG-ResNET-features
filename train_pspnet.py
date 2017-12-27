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

pspnet=build_pspnet( 50, input_shape)

pspnet.summary()


