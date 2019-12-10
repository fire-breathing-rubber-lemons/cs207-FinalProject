import numpy as np
import matplotlib.pyplot as plt

from pyad.nn import NeuralNet
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

np.random.seed(0)

X, y = load_boston(return_X_y=True)
X_scaled = preprocessing.scale(X)
y_scaled = preprocessing.scale(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=0
)


nn = NeuralNet(loss_fn='mse')
nn.add_layer(X_train.shape[1], 20, activation='linear')
nn.add_layer(20, 20, activation='relu')
nn.add_layer(20, 1, activation='linear')


print('Pre-train loss on train data:', nn.score(X_train, y_train).value)
print('Pre-train loss on test data:', nn.score(X_test, y_test).value)

epochs = [0]
train_loss = [nn.score(X_train, y_train).value]
test_loss = [nn.score(X_test, y_test).value]

for i in range(100):
    nn.train(
        X_train, y_train, X_test, y_test,
        batch_size=20, epochs=1, learning_rate=1e-1, verbose=False
    )
    epochs.append(i)
    train_loss.append(nn.score(X_train, y_train).value)
    test_loss.append(nn.score(X_test, y_test).value)

    if (i + 1) % 10 == 0:
        print(f'{i + 1}/100 loops completed')

plt.plot(epochs, train_loss)
plt.plot(epochs, test_loss)

plt.title('Loss over time')
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Loss (mean squared error)')

print('\nFinal loss on train data:', nn.score(X_train, y_train).value)
print('Final loss on test data:', nn.score(X_test, y_test).value)


def compute_r2(x, y):
    predictions = nn.predict(x)
    tss = np.sum((y - np.mean(y)) ** 2)
    ess = np.sum((y - predictions) ** 2)
    return 1 - ess/tss


print('\nFinal R^2 on train data:', compute_r2(X_train, y_train))
print('Final R^2 on test data:', compute_r2(X_test, y_test))
plt.show()
