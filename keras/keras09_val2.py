from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

# 1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

x_test = np.array([16, 17, 18, 19, 20])
y_test = np.array([16, 17, 18, 19, 20])

# x_train = np.array([1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15])
# y_train = np.array([1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15])

# 2. 모델 구성
model = Sequential()

model.add(Dense(3, input_dim=1))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_test, y_test, epochs=100, validation_split=0.2)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

model.summary()
