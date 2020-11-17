from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

dataset = load_diabetes()

x = dataset.data.reshape(dataset.data.shape[0], dataset.data.shape[1], 1)
y = dataset.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# 2. 모델구성
model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(x.shape[1], 1)))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()

es = EarlyStopping(monitor='loss', patience=25, mode='auto')
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=1000, batch_size=32,
                    verbose=1, validation_split=0.2, callbacks=[es, to_hist])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss : ", loss)

y_predict = model.predict(x_test)
# R2
r2 = r2_score(y_test, y_predict)

print("R2 : ", r2)
