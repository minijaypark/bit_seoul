# One Hot Encoding

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32 * 32 * 3).astype('float32')/255
x_test = x_test.reshape(10000, 32 * 32 * 3).astype('float32')/255


model = Sequential()
model.add(Dense(256, input_shape=(32 * 32 * 3,)))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(10, activation='softmax'))

model.summary()

es = EarlyStopping(monitor='loss', patience=5, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=100, batch_size=32,
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
