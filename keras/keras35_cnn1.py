from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(10, (2, 2), input_shape=(10, 10, 1)))      # Output: 9 9 10
model.add(Conv2D(5, (2, 2), padding='same'))                # Output: 9 9 5
model.add(Conv2D(3, (3, 3), padding='valid'))               # Output: 7 7 3
model.add(Conv2D(7, (2, 2)))                                # Output: 6 6 7

'''
Conv2D

number_parameters = out_channels * (in_channels * kernel_h * kernel_w + 1)  # 1 for bias
'''
model.add(MaxPool2D())                                      # Output: 3 3 7
model.add(Flatten())                                        # Output: 63
model.add(Dense(1))

model.summary()
