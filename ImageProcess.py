# -*- coding: utf-8 -*-
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "data/train.p"
validation_file= "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import csv

# Example data
all_labels = []

#Select workbook
with open('signnames.csv', 'r') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        all_labels += [row[1]]

all_labels = all_labels[:-1]
# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape
img_size=X_train.shape[1]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(all_labels)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import random
import cv2
from itertools import groupby
# %matplotlib inline

### Show images with it label.
num_of_samples=[]
plt.figure(figsize=(15, 30))
for i in range(0, n_classes):
    plt.subplot(15, 3, i+1)
    x_selected = X_train[y_train == i]
    img = x_selected[0]
    plt.imshow(img) #draw the first image of each class
    plt.title(all_labels[i])
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()

### Show number of images per label
fig, ax = plt.subplots(figsize=(10, 15))

labels = np.arange(len(all_labels))
numOfEach = [len(list(group)) for key, group in groupby(y_train)]
avgOfAll = np.mean(numOfEach)
print("Average number of training images:", int(avgOfAll))
ax.barh(labels, numOfEach, align='center', color='blue')
ax.set_yticks(labels)
ax.set_yticklabels(all_labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of Training images')
ax.set_title('Number of Training images for each sign')
plt.show()

class ImageEffect:

    def transformImage(self, img, effects):
        if effects == 0:
            img = self.translate(img)
        if effects == 1:
            img = self.rotate(img)
        if effects == 2:
            img = self.shear(img)
        if effects == 3:
            img = self.blur(img)
        if effects == 4:
            img = self.gamma(img)
        if effects == 5:
            img = self.flip(img)
        if effects == 6:
            img = self.pad(img)
        return img


    def translate(self, img):
        x = img.shape[0]
        y = img.shape[1]

        x_shift = np.random.uniform(-0.3 * x, 0.3 * x)
        y_shift = np.random.uniform(-0.3 * y, 0.3 * y)

        shift_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shift_img = cv2.warpAffine(img, shift_matrix, (x, y))

        return shift_img


    def rotate(self, img):
        row, col, channel = img.shape

        angle = np.random.uniform(-60, 60)
        rotation_point = (row / 2, col / 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_point, angle, 1)

        rotated_img = cv2.warpAffine(img, rotation_matrix, (col, row))
        return rotated_img
    
    
    def shear(self, img):
        x, y, channel = img.shape

        shear = np.random.randint(5,15)
        pts1 = np.array([[5, 5], [20, 5], [5, 20]]).astype('float32')
        pt1 = 5 + shear * np.random.uniform() - shear / 2
        pt2 = 20 + shear * np.random.uniform() - shear / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

        M = cv2.getAffineTransform(pts1, pts2)
        result = cv2.warpAffine(img, M, (y, x))
        return result


    def blur(self, img):
        r_int = np.random.randint(0, 2)
        odd_size = 2 * r_int + 1
        return cv2.GaussianBlur(img, (odd_size, odd_size), 0)


    def gamma(self, img):
        gamma = np.random.uniform(0.3, 1.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_img = cv2.LUT(img, table)
        return new_img
    
    
    def flip(self, img):
        lr = bool(random.getrandbits(1))
        ud = bool(random.getrandbits(1))
        if lr:
            img = cv2.flip(img, flipCode=1)
        if ud:
            img = cv2.flip(img, flipCode=0)
        return img
    
    
    def pad(self, img, pad_width=None, axis=0, mode='symmetric'):
        hei,wid=img.shape[0],img.shape[1]      
        if pad_width is None:
            th=hei//10
            tw=wid//10
            pad_width=((th,th),(tw,tw),(0,0))
        if axis==0:
            if type(pad_width[0])==tuple:
                pad_width=(pad_width[0],(0,0),(0,0))
            else:
                pad_width=(pad_width,(0,0),(0,0))
        if axis==1:
            if type(pad_width[0])==tuple:
                pad_width=((0,0),pad_width[1],(0,0))
            else:
                pad_width=((0,0),pad_width,(0,0))
        if len(img.shape)==3:
            newimage=np.pad(img,pad_width,mode)
        elif len(img.shape)==2:
            newimage=np.squeeze(np.pad(img[:,:,np.newaxis],pad_width,mode))
        
        return cv2.resize(newimage,(wid,hei),interpolation=cv2.INTER_NEAREST)
    

### Show images with it label.
num_of_samples=[]
plt.figure(figsize=(15, 30))
for i in range(0, n_classes):
    plt.subplot(15, 3, i+1)
    x_selected = X_train[y_train == i]
    imgEf = ImageEffect()
    img = x_selected[0]
    img = imgEf.transformImage(img, 5)
    plt.imshow(img) #draw the first image of each class
    plt.title(all_labels[i])
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()