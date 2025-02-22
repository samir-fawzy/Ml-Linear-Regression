import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = "C:/ML/Ml-Linear-Regression/data.txt"
data = pd.read_csv(path, header=None, names=["Population", "Profit"])

data.insert(0,"Bais",1)
cols = data.shape[1]

x = data.iloc[ : , : cols - 1]
y = data.iloc[ : ,cols -1 : cols]
w = np.zeros((1,x.shape[1]))


x = np.matrix(x.values)
y = np.matrix(y.values)

def LinearFunc(x,theta):
    return np.dot(x , theta)

def ComputeCost(x,y,theta):
    loss = np.power((y - LinearFunc(x,theta)),2)
    return np.sum(loss) / (2 * len(y))

def GradientDescent(x,y,theta,alpha,iters):
    m = len(y)
    for i in range(iters):
        gradient = x.T @ (LinearFunc(x,theta) - y)  / m
        theta -= alpha * gradient 
    return theta

alpha = 0.001
iters = 100000

print("Mean Cost = ",ComputeCost(x,y,w.T))

w = GradientDescent(x,y,w.T,alpha,iters)
print("new theta = ",w)

print("Cost = ",ComputeCost(x,y,w))

print("Prediction = ",LinearFunc(x,w))

x_line = np.linspace(data.Population.min(),data.Population.max(),100) 
y_ = w[0,0] + w[1,0] * x_line

fig,ax = plt.subplots(figsize=(12,8))
ax.scatter(data.Population,data.Profit,label="Training Data")
ax.plot(x_line,y_,"r",label="Prediction")
ax.legend(loc=2)
ax.set_xlabel("Population")
ax.set_ylabel("Profit")
ax.set_title("Predicted Profit vs Population Size")
plt.show()