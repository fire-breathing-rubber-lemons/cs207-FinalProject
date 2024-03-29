import numpy as np
from pyad.nn import NeuralNet
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

np.random.seed(0)
data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, train_size=0.8, random_state=0
)

nn = NeuralNet(loss_fn='cross_entropy')
nn.add_layer(X_train.shape[1], 100, activation='linear')
nn.add_layer(100, 100, activation='logistic')
nn.add_layer(100, 1 + np.max(y_train), activation='linear')

nn.train(
    X_train, y_train, X_test, y_test,
    batch_size=1, learning_rate=1e-3, epochs=20
)


print('Predictions:', nn.predict(X_test))
