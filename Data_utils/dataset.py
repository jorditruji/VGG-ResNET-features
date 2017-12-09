import numpy as np
from itertools import islice
import subprocess
import os, sys
import numpy as np
import time
import shutil
from keras.utils import np_utils
import random
import cv2
from scipy import misc, ndimage, io
import re
from itertools import product




def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def read_image(name):
    img=misc.imread(name)
    img_Resize= misc.imresize(img, (640, 480))

    return img_Resize

def read_label(name):
    label=io.loadmat(name)[name[:-4]]
    #label = label.ravel()
    #label = np_utils.to_categorical(label, 16)
    #print (label.shape)
    return label



def equalize_hist(img):
    equ = cv2.equalizeHist(img)
    return equ



def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def img2int(img):
    zmax=np.max(img)
    norm_img=np.zeros(img.shape,dtype=np.uint8)
    mask=np.zeros(img.shape,dtype=np.uint8)
    mask_std=np.zeros(img.shape,dtype=np.uint8)
    cont=0
    h,w = img.shape
    for pos in product(range(h), range(w)):
        pixel =  img.item(pos[0],pos[1])
        if pixel>0:
            new_pix2=(float(pixel)/float(zmax))*254.0
            norm_img[pos]=255-new_pix2
            mask[pos]=0
        else:
            norm_img[pos]=0
            if (pos[1]>20 and pos [0]>20):
                mask[pos]=255
            else:
                mask[pos]=255
        cont+=1

    dst_TELEA = cv2.inpaint(norm_img,mask,3,cv2.INPAINT_TELEA)
    dst_TELEA=equalize_hist(dst_TELEA)
    return dst_TELEA


def create_mean(path):
    filename = path
    images =[]
    labels=[]
    with open(filename) as f:
        for line in f:
            prova =line.strip().split(' ')

            images.append(calc_mean(read_image(prova[0])))

    print (sum(images) / float(len(images)))
    return sum(images) / float(len(images))


def calc_mean(image):

    return np.mean(image, axis=(0, 1))




class dataset:
    """
        Class Dataset:
            This class contains data generators that will feed the training of our models
            - Parameters:
                - batch_size => number of images to delivered every batch
                - samples_train => total number of images on training_dataset
                - samples_val => total number of images on validation_dataset

    """

    def __init__(self, batch_size, samples_train, samples_val):
        self.samples_train = samples_train
        self.samples_val = samples_val
        self.batch_size = batch_size
        self.data_mean = np.array([[[126.92261499, 114.11585906, 99.15394194]]])  # RGB order

    def train_generator(self, filename):
        while True:
            images =[]
            labels=[]
            i=0

            with open(filename) as f:
                head = list(islice(f, self.samples_train))
            
                for line in head:
                    #printProgressBar(i + 1, len(head), prefix='Progress:', suffix='Complete', length=50)
                    i += 1
                    file_names =line.strip().split(' ')
                    img=read_image(file_names[0])
                    float_img = img.astype('float16')
                    centered_image = float_img - self.data_mean
                    bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
                    input_data = bgr_image[np.newaxis, :, :, :] 
                    images.append(input_data)
                    label=file_names[1]
                    labels.append(img2int(read_pgm(label[:-4],'>')))

                    if i%self.batch_size==0:
                        images=np.array(images)
                        labels=np.array(labels)
                        print (labels.shape)
                        labels=labels.reshape((self.batch_size, 640, 480, 1))
                        print (labels.shape)
                        images= np.squeeze(images)
                        yield images, labels
                        images = []
                        labels = []

    def val_generator(self, filename):
        while True:
            images =[]
            labels=[]
            i=0

            with open(filename) as f:
                head = list(islice(f, self.samples_val))
            
                for line in head:
                    #printProgressBar(i + 1, len(head), prefix='Progress:', suffix='Complete', length=50)
                    i += 1
                    file_names =line.strip().split(' ')
                    img=read_image(file_names[0])
                    float_img = img.astype('float16')
                    centered_image = float_img - DATA_MEAN
                    bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
                    input_data = bgr_image[np.newaxis, :, :, :] 
                    images.append(input_data)
                    label=file_names[1]
                    labels.append(img2int(read_pgm(label[:-4],'>')))

                    if i%self.batch_size==0:
                        images=np.array(images)
                        labels=np.array(labels)
                        labels=labels.reshape((self.batch_size, 640, 480, 1))
                        images= np.squeeze(images)
                        yield images, labels
                        images = []
                        labels = []


