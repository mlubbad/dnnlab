import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import datasets, models, layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Flatten
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random

import keras
from keras.models import Sequential
from keras.layers import Dense

#plt.figure(figsize=(20,20))
#img_folder=r'/home/server/muhammed/training/straumann'
#os.listdir(img_folder)

#for i in range(5):
#    file = random.choice(os.listdir(img_folder))
#    image_path= os.path.join(img_folder, file)
#    img=mpimg.imread(image_path)
#    ax=plt.subplot(1,5,i+1)
#    ax.title.set_text(file)
#    plt.imshow(img)

IMG_WIDTH=200
IMG_HEIGHT=100
train_img_folder=r'/home/server/muhammed/training/'
test_img_folder=r'/home/server/muhammed/testing/'

def create_dataset(img_folder):
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
# extract the image array and class name
train_images, class_name_train =create_dataset(train_img_folder)
test_images, class_name_test  =create_dataset(test_img_folder)

target_dict={k: v for v, k in enumerate(np.unique(class_name_train))}
target_dict

train_labels =  [target_dict[class_name_train[i]] for i in range(len(class_name_train))]
test_labels =  [target_dict[class_name_test[i]] for i in range(len(class_name_test))]

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1

#train_images, test_images = train_images / 255.0, test_images / 255.0

#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck']

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i])
#    # The CIFAR labels happen to be arrays, 
#    # which is why you need the extra index
#    plt.xlabel(class_name_train[train_labels[i]])
#plt.show()

model=tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(6)
        ])
#model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

batch_size=32
#history = model.fit(x=tf.cast(np.array(train_images), tf.float64), y=tf.cast(list(map(int,train_labels)),tf.int32), epochs=10)
history = model.fit(x=np.array(train_images, np.float32), y=np.array(list(map(int,train_labels)), np.float32), epochs=50,batch_size=32, validation_data=(np.array(test_images, np.float32), np.array(list(map(int,test_labels)), np.float32)), verbose=2)
#history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images, test_labels))

# Save the entire model as a SavedModel.
model.save('/home/server/muhammed/saved_model/cnn_model')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(np.array(test_images, np.float32), np.array(list(map(int,test_labels)), np.float32), verbose=2)

print(test_acc)

plt.show()
