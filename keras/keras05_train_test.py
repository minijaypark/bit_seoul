from tensorflow.keras.models import Sequential # tf 안에 keras 안에 model에 seq를 가져온다
from tensorflow.keras.layers import Dense,Activation # tf 안에 keras 안에 layers에 Dense를 가져온다.
from sklearn.metrics import mean_squared_error
from keras.optimizers import SGD
import numpy as np

# 1. 데이터 준비
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])

# 2. 모델구성
# DNN을 Dense층으로 구성한다.
# input_dim : 입력 뉴런의 수를 설정합니다.
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile( loss='mse', optimizer='adam', metrics=['mae'])
# model.compile(loss='mse', optimizer='adam')
# compile구간에서 선언하지않으면 출력하지 않는다.

# mse : Mean Squared Error : 손실함수 : 평균 제곱 오차 : 정답에 대한 오류를 숫자로 나타내는 것
# optimizer(최적화)를 adam으로 사용하겠다.
# metrics : 평가지표 
# acc : accuracy : 정확성

model.fit(x,y, epochs=100, batch_size=1)
# model.fit(x,y, epochs=100)

# model.fit : 이 모델을 훈련시키겠다.
# epochs : 몇번 훈련시키겠다.
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정 
# batch_size default : 32
# epochs default : 100


# 4. 평가, 예측
# loss, acc = model.evaluate(x,y,batch_size=1)
loss = model.evaluate(x,y, batch_size=1)

# loss, acc = model.evaluate(x,y)
# loss : 훈련 손실값 acc : 훈련 정확도
# val_loss : 검증 손실값 val_acc : 검증 정확도

print("loss : ",loss)
# print("acc : ",acc)
# accuracy : 맞췄다 못맞췄다에 대한 값


x_pred = model.predict(x_pred)
print("x_pred : \n",x_pred)






