import pandas as pd
import numpy as np
import random

def sigmoid(x):
	return 1/(1+np.exp(-x))


def gradient(X,y,Theta):
	m=len(y[:,1])
	Z=(sigmoid(X.dot(Theta))-y)/m
	return (X.T).dot(Z)


def gradientDecent(X,y,Theta,epochs,alpha=0.15):
	i=0
	while(i<epochs):
		print("Iteration:" + str(i+1))
		Theta=Theta-alpha*gradient(X,y,Theta)
		i+=1
		print(Costfuncn(X,y,Theta))
	return Theta


def Costfuncn(X,y,Theta):
	m=len(y)
	subtract=np.array([[1,1,1]]*m)
	#print(np.shape(subtract))
	cost1=np.log(sigmoid(X.dot(Theta))).T
	cost2=np.log(subtract-sigmoid(X.dot(Theta))).T
	return sum(sum(-y.dot(cost1)-(subtract-y).dot(cost2)))



traindata=pd.read_csv("iris.data")
testdata=pd.read_csv("iristest.data")
Xtest=np.array(testdata)
ytest=Xtest[:,-1]
Xtest=Xtest[:,0:4]
Xfull=np.array(traindata)
yfull=Xfull[:,-1]
Xfull=Xfull[:,0:4]
bias1=[[1]]*len(Xfull[:,0])
bias2=[[1]]*len(Xtest[:,0])
Xfull=np.concatenate((bias1,Xfull),axis=1)
Xtest=np.concatenate((bias2,Xtest),axis=1)
Xfull=np.array(Xfull,dtype="float64")
Xtest=np.array(Xtest,dtype="float64")
#print(Xfull)
#print(Xtest)


y=[]
for i in range(len(yfull)):
	if yfull[i] == 'Iris-setosa':
		q=[1,0,0]
		y.append(q)
	elif yfull[i] == 'Iris-versicolor':
		q=[0,1,0]
		y.append(q)
	elif yfull[i] == 'Iris-virginica':
		q=[0,0,1]
		y.append(q)
y=np.array(y,dtype="float64")
#print(y)

Theta=np.random.rand(5,3)
print("Learning.....")
ThetaNew=gradientDecent(Xfull,y,Theta,1000)
predictions=sigmoid(Xtest.dot(ThetaNew))
predictclass=np.argmax(predictions,axis=1)
model_predictions=[]
for i in range(len(predictclass)):
	if predictclass[i]==0:
		model_predictions.append('Iris-setosa')
	elif predictclass[i]==1:
		model_predictions.append('Iris-versicolor')
	elif predictclass[i]==2:
		model_predictions.append('Iris-virginica')
	
	
print("=============================================")
print("Actual")
print(ytest)
print("=============================================")
print("Model Prediction")
print(model_predictions)
