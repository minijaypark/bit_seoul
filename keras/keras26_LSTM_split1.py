import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

dataset = range(1, 11)
size = 5


def split_x(seq, size):
    result = []
    for i in range(len(seq) - size + 1):
        subset = seq[i: (i+size)]
        result.append([item for item in subset])
    return np.array(result)


x = np.array(split_x(dataset, size))
y = x[:, 4]
x = x[:, :4]

x_test = np.array([[7, 8, 9, 10]])
# y_test = np.array([11])

# #2. model
input = Input(shape=(3, 1))
lstm = LSTM(200)(input)
dense = Dense(10)(lstm)
dense = Dense(20)(dense)
output = Dense(1)(dense)

model = Model(inputs=[input], outputs=[output])
model.summary()

# #Compile
model.compile(loss='mse', metrics=[], optimizer='adam')
loss = model.fit(x, y, batch_size=32, epochs=1000)

# #predict
# loss = model.evaluate(x_test, y_test, batch_size=32)
y_predict = model.predict(x_test)

print(y_predict)
