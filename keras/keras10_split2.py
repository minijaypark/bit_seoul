from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. data
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]    # 60개
y_train = y[:60]

x_val = x[61:81]   # 20개
y_val = y[61:81]

x_test = x[81:101]  # 20개
y_test = y[81:101]

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
