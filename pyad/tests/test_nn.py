import pyad.reverse_mode as rev
import pyad.nn as nn
import numpy as np
import pytest


def test_activations():
    """
    Test the activation functions values. Their derivatives should already be
    correct assuming the underlying primitive operations are correct
    """

    x = rev.Tensor([-1, 2, 3])
    assert nn.linear_activation(x) is x

    softmax = nn.softmax_activation(x)
    assert np.allclose(softmax.value, [0.01321289, 0.26538793, 0.72139918])

    log_softmax = nn.log_softmax_activation(x)
    assert np.allclose(log_softmax.value, [-4.32656264, -1.32656264, -0.32656264])

    tanh = nn.tanh_activation(x)
    assert np.allclose(tanh.value, [-0.76159416,  0.96402758,  0.99505475])

    logistic = nn.logistic_activation(x)
    assert np.allclose(logistic.value, [0.26894142, 0.88079708, 0.95257413])

    relu = nn.relu_activation(x)
    assert np.allclose(relu.value, [0, 2, 3])


def test_regression_loss_functions():
    """
    Test the regression loss functions to make sure their values are correct.
    """
    y_pred = rev.Tensor([-3, 5, 6])
    y_true = rev.Tensor([-5, 1, 7])

    mse = nn.mse_loss(y_pred, y_true)
    assert np.isclose(mse.value, 7.0)

    rmse = nn.rmse_loss(y_pred, y_true)
    assert np.isclose(rmse.value, 7.0 ** 0.5)


def test_classification_loss_functions():
    """
    Test the classification loss functions to make sure their values are correct.
    """
    y_pred = rev.Tensor([[-3, 4, 1], [-2, 5, 3]])
    y_true = rev.Tensor([0, 1, 1])

    nll = nn.nll_loss(y_pred, y_true)
    assert np.isclose(nll.value, -5/3)

    cross_entropy = nn.cross_entropy_loss(y_pred, y_true)
    assert np.isclose(cross_entropy.value, 0.5844837953598061)


def test_regression_NeuralNet():
    """
    Test building and testing a regression NeuralNet
    """

    # try out an invalid loss function
    with pytest.raises(ValueError):
        nn.NeuralNet(loss_fn='blarg')

    net = nn.NeuralNet('mse')

    # try adding an invalid activation
    with pytest.raises(ValueError):
        net.add_layer(10, 10, activation='blarg')

    # add a single layer. hardcode its weights
    net.add_layer(3, 1, activation='linear')
    net.layers[0]['weights'] = rev.Tensor([[1, 2, 3]])
    net.layers[0]['bias'] = rev.Tensor([[1]])

    # try adding an improperly sized layer
    with pytest.raises(ValueError):
        net.add_layer(19, 20, activation='linear')

    # evaluate
    x = np.array([[3, 2, 1], [4, 5, 6]])
    y = np.array([6, 10])
    res = net.evaluate(x)

    assert np.allclose(res.value, [[11, 33]])
    assert net.score(x, y).value == 277

    # predict
    assert np.allclose(net.predict(x), [[11, 33]])

    # accuracy
    with pytest.raises(ValueError):
        net.accuracy(x, y)


def test_classification_NeuralNet():
    """
    Test building and testing a classficiation NeuralNet
    """

    net = nn.NeuralNet(loss_fn='nll')

    # add a singel layer. hardcode its weights
    net.add_layer(3, 2, activation='log_softmax')
    net.layers[0]['weights'] = rev.Tensor([[1, 2, 3], [4, 5, 6]])
    net.layers[0]['bias'] = rev.Tensor([[1], [2]])

    # evaluate
    x = np.array([[3, 2, 1], [4, 5, 6]])
    y = np.array([0, 1])
    res = net.evaluate(x)

    assert np.allclose(res.value, [[-19, -46], [-5.60279645e-09, 0]])
    assert np.isclose(net.score(x, y).value, 9.500000002801398)

    # predict
    assert np.allclose(net.predict(x), [1, 1])

    # accuracy
    assert net.accuracy(x, y) == 0.5


def test_train_NeuralNet():
    """
    Test training a NeuralNet on a simple dataset with 2 features and 2 classes
    """

    np.random.seed(0)
    net = nn.NeuralNet(loss_fn='nll')
    net.add_layer(2, 30, activation='relu')
    net.add_layer(30, 2, activation='log_softmax')

    # generate a classic XOR dataset
    x = np.random.random((250, 2)) - 0.5
    y = (np.sign(x[:, 0]) != np.sign(x[:, 1])).astype(np.int64)

    split = int(len(x) * 0.8)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    # train it
    net.train(
        x_train, y_train, x_val, y_val,
        batch_size=2, epochs=10, learning_rate=1e-1
    )

    # check that the losses are low and the accuracies are high
    assert net.score(x_train, y_train).value < 0.5
    assert net.score(x_val, y_val).value < 0.5

    assert net.accuracy(x_train, y_train) >= 0.9
    assert net.accuracy(x_val, y_val) >= 0.9
