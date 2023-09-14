import os
import cv2
import numpy as np
from tensorflow import keras
from keras import layers
from google.colab.patches import cv2_imshow
from cv2 import imwrite
import scipy.io as sio
# Set up paths to the dataset directories
path_to_prid_dataset = '/content/drive/MyDrive/prid_2011'
path_to_camera_a = '/content/drive/MyDrive/prid_2011/single_shot/cam_a'
path_to_camera_b = '/content/drive/MyDrive/prid_2011/single_shot/cam_b'
# Load and preprocess the images
def load_and_preprocess_image(image_path):
    #print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    return image.reshape((256, 256, 1))
images_camera_a = [load_and_preprocess_image(os.path.join(path_to_camera_a, filename)) for filename in os.listdir(path_to_camera_a)]
images_camera_b = [load_and_preprocess_image(os.path.join(path_to_camera_b, filename)) for filename in os.listdir(path_to_camera_b)]
# Prepare the data
train_images_a = images_camera_a[:89]
train_images_b = images_camera_b[:89]
labels=sio.loadmat('/content/Graph_Cost.mat')
labels=labels['Graph_Cost']
train_set = []
train_labels = []
for i in range(62):
    for j in range(62):
        train_set.append(np.concatenate((train_images_a[i],train_images_b[j])))
        train_labels.append(labels[i][j]/100)
train_set = np.array(train_set)
train_labels = np.array(train_labels)
val_set = []
val_labels=[]
for i in range(62,89,1):
    for j in range(62,89,1):
        val_set.append(np.concatenate((train_images_a[i],train_images_b[j])))
        val_labels.append(labels[i][j]/100)
val_set = np.array(val_set)
val_labels = np.array(val_labels)