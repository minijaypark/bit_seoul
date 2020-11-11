import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Dense, Input, concatenate
from tensorflow.keras.models import Model, Sequential

# Data Set

x1 = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
y1 = np.array([range(101, 201), range(711, 811), range(100)]).transpose()

x2 = np.array([range(1, 101), range(311, 411), range(100)]).transpose()
y2 = np.array([range(501, 601), range(711, 811), range(100)]).transpose()

x1_train, x1_test, y1_train, y1_test, x2_train, x2_test, y2_train, y2_test = train_test_split(
    x1, y1, x2, y2, train_size=0.7)

# Modeling

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(15, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
dense4 = Dense(5, activation='relu')(dense3)
output10 = Dense(1)(dense4)

# model1 = Model(inputs=input1, outputs=output1)

input2 = Input(shape=(3,))
dense5 = Dense(10, activation='relu')(input2)
dense6 = Dense(15, activation='relu')(dense5)
dense7 = Dense(10, activation='relu')(dense6)
output20 = Dense(1)(dense7)

# model2 = Model(inputs=input2, outputs=output2)

# model1.summary()
# model2.summary()

merge1 = Concatenate()([output10, output20])

middle1 = Dense(30)(merge1)
middle2 = Dense(7)(middle1)
middle3 = Dense(11)(middle2)

# output model 분기

output1 = Dense(30)(middle3)
output2 = Dense(7)(output1)
output3 = Dense(3)(output2)

output4 = Dense(15)(middle3)
output5 = Dense(14)(output4)
output6 = Dense(11)(output5)
output7 = Dense(3)(output6)

# define model

model = Model(inputs=[input1, input2], outputs=[output3, output7])
model.summary()

# Compile

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y1_train, y2_train],
          epochs=100, batch_size=8, validation_split=0.25)
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=8)

print("result: ", result)
