from tensorflow.keras.models import Sequential # tf 안에 keras 안에 model에 seq를 가져온다
from tensorflow.keras.layers import Dense,Activation # tf 안에 keras 안에 layers에 Dense를 가져온다.
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
import numpy as np

# 1. 데이터 준비
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

# 2. 모델구성
# DNN을 Dense층으로 구성한다.
# input_dim : 입력 뉴런의 수를 설정합니다.
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))
# 3개 layer 20 20 20

# 3. 컴파일, 훈련
model.compile( loss='mse', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adam')
# compile구간에서 선언하지않으면 출력하지 않는다.

# mse : Mean Squared Error : 손실함수 : 평균 제곱 오차 : 정답에 대한 오류를 숫자로 나타내는 것
# optimizer(최적화)를 adam으로 사용하겠다.
# metrics : 평가지표 
# acc : accuracy : 정확성

model.fit(x_train,y_train, epochs=100, batch_size=1)
# model.fit(x,y, epochs=100)

# model.fit : 이 모델을 훈련시키겠다.
# epochs : 몇번 훈련시키겠다.
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정 
# batch_size default : 32
# epochs default : 100


# 4. 평가, 예측
# loss, acc = model.evaluate(x,y,batch_size=1)
loss = model.evaluate(x_test,y_test, batch_size=1)

# loss, acc = model.evaluate(x,y)
# loss : 훈련 손실값 acc : 훈련 정확도
# val_loss : 검증 손실값 val_acc : 검증 정확도

print("loss : ",loss)
# print("acc : ",acc)
# accuracy : 맞췄다 못맞췄다에 대한 값

# 테스트값을 잘 예측했는지 평가해본다.
y_predict = model.predict(x_test)
print("y_predict : \n",y_predict)


# 실습 : 결과물 오차 수정. 미세조정

from sklearn.metrics import mean_squared_error

def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))