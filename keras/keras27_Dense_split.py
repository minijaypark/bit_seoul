import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential

dataset = range(1, 101)
size = 5


def split_x(seq, size):
    result = []
    for i in range(len(seq) - size):
        subset = seq[i: (i+size)]
        result.append([item for item in subset])
    return np.array(result)


x = np.array(split_x(dataset, size - 1))
y = np.array(dataset[size - 1:])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7)

# #2. model
model = Sequential()

model.add(Dense(3, input_shape=(size-1,)))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

# #Compile
model.compile(loss='mse', metrics=['acc'], optimizer='adam')
earlyStopping = EarlyStopping(monitor='loss', patience=125, mode='auto')
loss = model.fit(x_train, y_train, batch_size=32,
                 epochs=1000, callbacks=[earlyStopping])

# predict
loss, mse = model.evaluate([x_test], [y_test])
print(loss, mse)

x_pred = np.array([97, 98, 99, 100]).reshape(4, 1)
y_predict = model.predict(x_pred)
print(y_predict)
