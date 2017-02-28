# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
with open('train_feat.pickle','rb') as f:
    X_train_label = pickle.load(f, encoding="latin-1")
    
img_scaled = X_train_label[3]
img = (1+img_scaled) * 255/2
img_rbg = img.astype(np.uint8)
img_gray = cv2.cvtColor(img_rbg, cv2.COLOR_BGR2GRAY)

cv2.imwrite('a.png',img_rbg)
cv2.imwrite('b.png',img_gray)


# cifar dataset
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
input_shape=X_train.shape[1:]
print(input_shape)

# X has dimension 32x32x3 and is already normalized
depth, height, width = 3, 32, 32
with open('train_feat.pickle','rb') as f:
    X_train_label = pickle.load(f, encoding="latin-1")
X_train = X_train_label.astype('float32')

with open('validation_feat.pickle','rb') as f:
    X_valid_label = pickle.load(f, encoding="latin-1")
X_valid = X_valid_label.astype('float32')

#Y has 10 classes and need one-hot encoding
with open('train_lab.pickle','rb') as f:
    Y_train_label = pickle.load(f, encoding="latin-1")
Y_train = np_utils.to_categorical(Y_train_label, 10)

with open('validation_lab.pickle','rb') as f:
    Y_valid_label = pickle.load(f, encoding="latin-1")
Y_valid = np_utils.to_categorical(Y_valid_label, 10)


