import reverse_mode as rev
from NeuralNet_Rev import NeuralNet
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True)
X_scaled = preprocessing.scale(X)
y_scaled=preprocessing.scale(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.5, random_state=123)

np.random.seed(1)
testnn=NeuralNet(13,6,2)
testnn.init_params()
  
print("Pre-train loss on train data: ",testnn.score(X_train,y_train))
print("Pre-train loss on test data: ",testnn.score(X_test,y_test))   

train_loss=[testnn.score(X_train,y_train)]
test_loss=[testnn.score(X_test,y_test)]
for i in range(10):
    testnn.train(100,X_train,y_train,alpha=.01,v=False)
    train_loss.append(testnn.score(X_train,y_train))
    test_loss.append(testnn.score(X_test,y_test))

epochs=[i*100 for i in range(11)]
plt.plot(epochs,train_loss)
plt.plot(epochs,test_loss)
plt.legend(['Train', 'Test'], loc='upper left')
plt.xlabel("Epochs")
plt.ylabel("Loss")

train_score = testnn.score(X_train,y_train)
test_score=testnn.score(X_test,y_test)
print("Final loss on train data: ",train_score)
print("Final loss on test data: ",test_score)   

tss = sum((y_test-np.mean(y_test))**2)
predictions = testnn.predict(X_test)
ess = sum((y_test-predictions)**2)


print("R-squared on test data:",1-ess/tss)

tss = sum((y_train-np.mean(y_train))**2)
predictions = testnn.predict(X_train)
ess = sum((y_train-predictions)**2)

print("R-squared on train data:",1-ess/tss)