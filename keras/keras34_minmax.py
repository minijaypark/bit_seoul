import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from keras.layers import LSTM, Dense, Input
from keras.models import Model, Sequential

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [
    9, 10, 11], [10, 11, 12], [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000], [100, 200, 300]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])

x_predict = np.array([55, 65, 75]).reshape(1, 3)
x_predict2 = np.array([6600, 6700, 6800])

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_predict = scaler.transform(x_predict)

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(50))
model.add(Dense(10))

model.summary()
# Compile
model.compile(loss='mse', metrics=['mse'], optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min')
model.fit(x, y, batch_size=3, epochs=1000,
          verbose=1, callbacks=[earlyStopping])


y_predict = model.predict(x)
print(y_predict)

# loss = model.evaluate(x_input, np.array([80]), batch_size=1)
# print(loss)
