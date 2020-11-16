# One Hot Encoding

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

model = Sequential()
model.add(Conv2D(10, (2, 2), padding="same", input_shape=(28, 28, 1)))
model.add(Conv2D(20, (2, 2)))
model.add(Conv2D(30, (2, 2)))
model.add(Conv2D(40, (2, 2)))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

es = EarlyStopping(monitor='loss', patience=125, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                    verbose=1, validation_split=0.2, callbacks=[es, to_hist])

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", loss)
print("acc:", acc)


y_predict = model.predict(x_test[:10])
print("y predicts: ")
print([np.argmax(y, axis=None, out=None) for y in y_predict])
print()
print("real y's")
print([np.argmax(y, axis=None, out=None) for y in y_test[:10]])

'''
실습 1. test 데이터를 10개 가져와서 predict 만들기
실습 2. es, tensorboard 넣기
'''

# graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('Loss and Accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()
