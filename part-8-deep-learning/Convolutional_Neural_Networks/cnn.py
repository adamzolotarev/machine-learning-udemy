# Convolutional Neural Network

# For image processing:
# conda install pillow


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#  32, 3, 3 - 32 feature detectors with 3x3 dimentions
# 32 is a common practice to start with, also higher number - higher cpu/gpu usage
# input_shape - we will convert all images to the same size
# the expected format of our images is colored image and
# 64x64 pixels (smaller so we can run it in cpu)
# activation function - 'relu' is used to ensure
# we don't have negative pixels in order to have
# non-linearility
classifier.add(Convolution2D(
    32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# 2x2 is recommended
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# Common practice is to choose the average between the number of input nodes and output nodes
# We have way too many input nodes, so need to pick
# something by experimenting (128 is a good common choice)
# This is the hidden layer:
classifier.add(Dense(output_dim=128, activation='relu'))
# This is the output layer:
# sigmoid is for binary outcome; otherwise could use softmax
# TODO: update Dense call to something like classifier.add(Dense(128, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))


# Compiling the CNN
# 'adam' is stochastic gradient discend algorithm
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator


# ImageDataGenerator augments the images without having to
# add more images and helps with over-fitting
# rescale is kinda mandatory anyway
# geometrical transformations: shear, zoom, horizontal
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Jus rescaling so they have values between 0 and 1
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the path name to use absolute path - doesn't work with ~.
dataset_path = "./dataset/"

# target_size - image size
# batch_size - size of the batches in which our random images
# will be included, after wich the weight will be updated
# class_mode - 'binary' because we only have 2 values (dog/cat)

training_set = train_datagen.flow_from_directory(dataset_path + 'training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(dataset_path + 'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# we have 8000 images in the training set (2000 in the test set)
# nb_epoch - how many times we'll run through our training set
# (original is 25, but changed to 2 for testing, cause it's slow)
# nb_val_samples - corresponds to the number of images in the test set

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=2,
                         validation_data=test_set,
                         nb_val_samples=2000)

# val_acc is the accuracy of the test set
