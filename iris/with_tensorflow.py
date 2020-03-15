import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random

traindata=pd.read_csv('iris.data')
testdata=pd.read_csv("iristest.data")
Xtest=np.array(testdata)
ytest=Xtest[:,-1]
Xtest=Xtest[:,0:4]
Xfull=np.array(traindata)
yfull=Xfull[:,-1]
Xfull=Xfull[:,0:4]

for i in range(len(yfull)):
	if yfull[i] == 'Iris-setosa':
		yfull[i]=0
	elif yfull[i] == 'Iris-versicolor':
		yfull[i]=1
	elif yfull[i] == 'Iris-virginica':
		yfull[i]=2

n=len(Xfull[1,:])
model=keras.Sequential([
keras.layers.Dense(1000,input_dim=n,activation='sigmoid'),
keras.layers.Dense(500,activation='sigmoid'),
keras.layers.Dense(100,activation='sigmoid'),
keras.layers.Dense(500,activation='sigmoid'),
keras.layers.Dense(5,activation='sigmoid'),
keras.layers.Dense(3,activation='softmax'),
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(Xfull,yfull,epochs=250)
predictions=model.predict(Xtest)
prediclass=np.argmax(predictions,axis=1)

model_pridictions=[]

for i in range(len(prediclass)):
	if prediclass[i]==0:
		model_pridictions.append('Iris-setosa')
	elif prediclass[i]==1:
		model_pridictions.append('Iris-versicolor')
	elif prediclass[i]==2:
		model_pridictions.append('Iris-virginica')

print("Actual")
print(ytest)
print("Model Prediction")
print(model_pridictions)
