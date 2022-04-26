import cv2 as cv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

start_time = datetime.datetime.now()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100)

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')
print(f'Duration: {datetime.datetime.now()-start_time}')

model.save('digits.model')
model = tf.keras.models.load_model('digits.model')

for x in range(0, 10):
    img = cv.imread(f'digits/{x}.png')[:, :, 0]
    # For further optimization in flexibility
    # if img not (28,28):
    #     img = cv.resize(img, (28, 28))
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'Result: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
