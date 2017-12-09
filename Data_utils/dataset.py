import numpy as np


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
                        labels=labels.reshape((num_img, 640, 480, 1))
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
                        print (labels.shape)
                        labels=labels.reshape((num_img, 640, 480, 1))
                        print (labels.shape)
                        images= np.squeeze(images)
                        yield images, labels
                        images = []
                        labels = []


