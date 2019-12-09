import numpy as np
from pyad.nn import NeuralNet
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, train_size=0.8, random_state=0
)

nn = NeuralNet(loss_fn='mse')
nn.add_layer(X_train.shape[1], 50, activation='linear')
nn.add_layer(50, 50, activation='tanh')
nn.add_layer(50, 50, activation='tanh')
nn.add_layer(50, 1, activation='linear')

nn.train(
    X_train, y_train, X_test, y_test,
    batch_size=200, learning_rate=1e-4, epochs=20
)


print('Predicitons:', nn.predict(X_test))
