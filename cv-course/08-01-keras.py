
# neurons take in the input, assign weight
# then all goes through activation function
# activation functions makes sense of the input
# sigmoid for gradual 0-1, tanh for gradual -1-1, relu max(0, z(wx+b))

# neural network: input layer > hidden layers > output layer 
# cost functions show how far off we are from the expected value
# cross entropy quite fast depending on diff betwen a and y

# this error is corrected with gradient descent and back prop
# gradient descent finds optiman weight values 
# back prop updates the whole network with those weights


import numpy as np;
from numpy import genfromtxt

# data = genfromtxt('data/bank_note_data.txt', delimiter=',')
data = genfromtxt('data/bank_note_data.txt', delimiter=',')

labels = data[:,4]
features = data[:,0:4]

X = features
y = labels

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaled_X_test = MinMaxScaler().fit_transform(X_test)
scaled_X_train = MinMaxScaler().fit_transform(X_train)

import tensorflow as tf
from keras.models import  Sequential
from keras.layers import  Dense
from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = Sequential()
# 8 Neurons, expects input of 4 features. 
# Play around with the number of neurons!!
model.add(Dense(4, input_dim=4, activation='relu'))
# Add another Densely Connected layer (every neuron connected to every neuron in the next layer)
model.add(Dense(8, activation='relu'))
# Last layer simple sigmoid function to output 0 or 1 (our label)
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(scaled_X_train,y_train,epochs=500, verbose=2)

model.predict_classes(scaled_X_test)
from sklearn.metrics import confusion_matrix, classification_report

preds = model.predict_classes(scaled_X_test)

print(classification_report(y_test, preds))

model.save('model_500.h5')

from keras.models import load_model
model = load_model('model_500.h5')
print(model.predict_classes(scaled_X_test))