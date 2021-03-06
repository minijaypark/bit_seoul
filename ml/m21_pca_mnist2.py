# 실습 
# PCA를 통해 마구마구 0.95 이상인게 몇개?
# 1 이상
# mnist DNN과 loss / acc 를 비교

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) #(60000, 28, 28)
print(x_test.shape) #(10000, 28, 28)

# x_train x_test를 append했으니 행이 늘어난다 그래서70000
x = np.append(x_train, x_test, axis=0)
print(x.shape) #(70000, 28, 28)

x = x.reshape(70000, 784)

'''
pca = PCA(n_components=154) # 내가 출력하고 싶은 열을 설정할 수 있다 차원축소
x2d = pca.fit_transform((x))
print(x2d.shape) #(70000, 154)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) #1.
print(sum(pca_EVR)) #2.

#2
#0.9481826884444571

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 0.95) +1
print(cumsum >= 0.95) 
print(d) #우리가 원하는 n_components의 개수
'''

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

'''
pca = PCA(n_components=713) # 내가 출력하고 싶은 열을 설정할 수 있다 차원축소
x2d = pca.fit_transform((x))
print(x2d.shape) #(70000, 9)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) #1.
print(sum(pca_EVR)) #2.
#2
#0.9481826884444571

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

d = np.argmax(cumsum >= 1) +1
print(cumsum >= 1) 
print(d) #우리가 원하는 n_components의 개수
'''

# PCA로 컬럼 걸러내기
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_) #누적된 합 표시

d = np.argmax(cumsum >= 0.95) +1
print("d값 : ", d) #우리가 원하는 n_components의 개수 154개

# pca1 변수 선언 d값을 n_components 대입, x값 트랜스폼
pca1 = PCA(n_components=d)
x = pca1.fit_transform(x)

x_train = x[:60000, :]
x_test = x[60000:, :]

#원핫 적용
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#이미지 조정
x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.

#2. 모델 구성
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(d, ))) #(70000,154)
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu')) # strides=2 니까 2칸씩 이동한다
model.add(Dense(20, activation='relu'))
model.add(Dense(100, activation='relu')) #Dense 디폴트는  linear
model.add(Dense(10, activation='softmax')) # 꼭 들어감

model.summary()

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
early_Stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=1000, verbose=1, validation_split=0.2, callbacks=[early_Stopping]) #원래 배치 사이즈의 디폴트는 32

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1000)

print("loss : ", loss)
print("acc : ", acc)


'''
loss :  0.21922236680984497
acc :  0.9453999996185303
'''
