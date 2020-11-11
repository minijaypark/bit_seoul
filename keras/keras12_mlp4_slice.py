from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

x = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
y = np.array([range(101, 201), range(711, 811), range(100)]).transpose()

x_train = x[:70]
x_test = x[70:100]
y_train = y[:70]
y_test = y[70:100]

model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(7))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

model.summary()

# R2
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
