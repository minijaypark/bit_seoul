import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Dense, Input, concatenate
from tensorflow.keras.models import Model, Sequential

# Data Set

x1 = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
x2 = np.array([range(1, 101), range(311, 411), range(100)]).transpose()

y1 = np.array([range(101, 201), range(711, 811), range(100)]).transpose()
y2 = np.array([range(501, 601), range(711, 811), range(100)]).transpose()
y3 = np.array([range(501, 601), range(711, 811), range(100)]).transpose()

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1, x2, y1, y2, y3, train_size=0.7)

# Modeling

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(15, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
dense1 = Dense(1)(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(15, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(1)(dense2)

merge = Concatenate()([dense1, dense2])
merge = Dense(30)(merge)
merge = Dense(7)(merge)
merge = Dense(11)(merge)

# output model branching

output1 = Dense(30)(merge)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge)
output2 = Dense(14)(output2)
output2 = Dense(11)(output2)
output2 = Dense(3)(output2)

output3 = Dense(15)(merge)
output3 = Dense(14)(output3)
output3 = Dense(11)(output3)
output3 = Dense(3)(output3)

# define model

model = Model(inputs=[input1, input2], outputs=[output1, output2, output3])
model.summary()

# Compile

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train],
          epochs=100, batch_size=8, validation_split=0.25)
result = model.evaluate([x1_test, x2_test], [
                        y1_test, y2_test, y3_test], batch_size=8)

print("result: ", result)

y_predict = np.array(model.predict([x1_test, x2_test])).flatten()
y_test = np.array([y1_test, y2_test, y3_test]).flatten()


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


print("RMSE : ", RMSE(y_test, y_predict))

# R2
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
