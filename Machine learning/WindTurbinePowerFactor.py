# first neural network with keras tutorial
from numpy import loadtxt
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



dataset = loadtxt('wind-turbine.csv', delimiter=',')
X = dataset[:,0]
y = dataset[:,1]/5000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

# define the keras model
optimizer = keras.optimizers.Adam(learning_rate=0.0005)

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer= optimizer, metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
# evaluate the keras model
accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy[1]*100))

y_predicted = model.predict(X_test)
plt.plot(X_test, y_predicted, 'o', color='r')
plt.scatter(X_test, y_test)
plt.show()



