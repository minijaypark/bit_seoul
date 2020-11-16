from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(100, input_shape=(2, 1)))
model.add(Dense(50))
model.add(Dense(10))

model.save("./save/keras30.h5")
