from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model
import numpy as np

x = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
y = np.array(range(101, 201)).transpose()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7)

# model = Sequential()
# model.add(Dense(10, input_shape=(3,), activation='relu'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(15, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(5, activation='relu')(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1, outputs=output1)
model.summary()
'''
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

# 4. 평가, 예측
# 테스트값을 잘 예측했는지 평가해본다.
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

y_predict = model.predict(x_test)

model.summary()

# # 회귀모델에서 검증에 사용되는 평가지표 : RMSE R2
# # RMSE


# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))


# print("RMSE : ", RMSE(y_test, y_predict))

# R2
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''
