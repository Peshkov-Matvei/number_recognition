from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import func
import model
import numpy as numpy


func.create_my_sess(0.5)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255.
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255.

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


model = model.my_model()
model.fit(train_images, train_labels, epochs=7, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Точность на тестовых данных', test_acc)

model.save('mnist.h5')