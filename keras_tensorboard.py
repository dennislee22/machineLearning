import keras
import time
import datetime
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras import models
from keras import layers

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Loading data
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()

# Reshaping data-Adding number of channels as 1 (Grayscale images)
train_images = train_images.reshape((train_images.shape[0],
									train_images.shape[1],
									train_images.shape[2], 1))

test_images = test_images.reshape((test_images.shape[0],
								test_images.shape[1],
								test_images.shape[2], 1))

# Scaling down pixel values
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# Encoding labels to a binary class matrix
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation ="relu",
							input_shape =(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation ="relu"))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation ="relu"))
model.add(layers.Dense(10, activation ="softmax"))

model.summary()

model.compile(optimizer ="rmsprop", loss ="categorical_crossentropy",
											metrics =['accuracy'])

val_images = train_images[:10000]
partial_images = train_images[10000:]
val_labels = y_train[:10000]
partial_labels = y_train[10000:]

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
										mode ="min", patience = 5,
										restore_best_weights = True)

log_dir = "logs/tsrfit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

start = time.time()
#with tf.device('/CPU:0'):
with tf.device('/GPU:0'):
  history = model.fit(partial_images, partial_labels, batch_size = 128,
					epochs = 5, validation_data =(val_images, val_labels),
					callbacks =[tensorboard_callback])
end = time.time()
print("Time Taken: {}".format(end - start))
