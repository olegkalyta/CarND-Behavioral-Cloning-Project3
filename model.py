import csv
import cv2
import numpy as np
from random import randint

lines = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2


for line in lines:
    if float(line[3]) < .03 and randint(0,9) > 5:
        continue

    centerImage = cv2.imread('./IMG/' + line[0].split('\\')[-1])
    leftImage = cv2.imread('./IMG/' + line[1].split('\\')[-1])
    rightImage = cv2.imread('./IMG/' + line[2].split('\\')[-1])

    images.append(centerImage)
    images.append(leftImage)
    images.append(rightImage)

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

    images.append(cv2.flip(centerImage, 1))
    images.append(cv2.flip(leftImage, 1))
    images.append(cv2.flip(rightImage, 1))

    measurements.append(-1 * measurement)
    measurements.append(-1 * measurement + 0.2)
    measurements.append(-1 * measurement - 0.2)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(80, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(SpatialDropout2D(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')
