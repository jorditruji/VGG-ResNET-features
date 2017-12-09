import subprocess
import os, sys
import numpy as np
from itertools import islice
import time
import shutil
from keras.utils import np_utils
import random
import cv2
import numpy as np
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


