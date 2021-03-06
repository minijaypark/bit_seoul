import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

dataset = np.array(range(1, 101))
size = 5


def split_x(seq, size):
    result = []
    for i in range(len(seq) - size):
        subset = seq[i: (i+size)]
        result.append([item for item in subset])
    return np.array(result)


x = np.array(split_x(dataset, size))

y = x[:, size - 1]
x = x[:, :size - 1]

x = np.reshape(x, (x.shape[0], x.shape[1], 1))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=False)

# # 2. model
model = Sequential()
model.add(LSTM(100, input_shape=(2, 1)))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(5, name="keras29_1"))
model.add(Dense(1, name="keras29_2"))

# Compile
model.compile(loss='mse', metrics=['mae'], optimizer='adam')
es = EarlyStopping(monitor='loss', patience=125, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)
history = model.fit(x_train, y_train, batch_size=32,
                    epochs=700, callbacks=[es, to_hist], validation_split=0.2)

# #predict
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print(loss, mse)

# graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['mae'])
# plt.plot(history.history['val_acc'])

plt.title('Loss and Accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')

plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()
