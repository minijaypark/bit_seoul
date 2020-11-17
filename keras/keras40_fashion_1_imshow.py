import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

plt.imshow(x_train[0], 'gray')
plt.show()
