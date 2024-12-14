import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
path = 'C:\\ML\\ML-Linear-Regression\\data.txt'
data = pd.read_csv(path,header=None,names=['Population' , 'Profit'])
#
#show data details
print('data :\n',data.head(10))
print ('*' * 40)
print('data describtion :\n',data.describe())
print ('*' * 40)
#
#draw data
data.plot(kind='scatter', x='Population', y='Profit')
plt.show()
#
#adding new column calles one in first column his value is 1
data.insert(0,'one',1)
print('new data :\n',data.head(10))
print ('*' * 40)
#
#separate x (training data) from y (target variable)
cols = data.shape[1] # data : 97 rows and 3 columns , shape[1] = 3
x = data.iloc[ : , : cols - 1]
y = data.iloc[ : ,cols -1 : cols]

print ('x data = \n',x.head(10))
print ('*' * 40)
print ('y data = \n',y.head(10))
print ('*' * 40)
#
#convert data from data frame to numpy matrix
x = np.matrix(x.values) # convert x 
y = np.matrix(y.values) # convert y 
theta = np.matrix(np.array([0,0]))
# print ('x matrix : \n',x)
# print ('x matrix shape = ',x.shape)
# print('Theta = \n',theta)
# print ('y matrix : \n',y)
# print ('y matrix shape',y.shape)
# print ('*' * 40)
# 
# cost function
def ComputeCost(x,y,theta):
    # cost function = 1/2m * summition (h(x) - y)²
    #h(x) => theta0 + theta1 * x
    #m number of rows
    z = np.power(((x * theta.T) - y),2)
    #number of rows = length x
    num_rows = len(x)
    #summition = sum(z)
    return sum(z) / (2 * num_rows)

print ('Compute Cost = ',ComputeCost(x, y, theta))
print ('*' * 40)
#
# GD => Gradiant Descent Function
# theta = theta - alpha * derivative compute cost 
def GradiantDescient(x,y,theta,alpha,iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (x * theta.T) - y
        
        for j in range(parameters): # 0 , 1
            term = np.multiply(error,x[:,j])
            # j = 0 => x = 1
            # j = 1 => x = value
            temp[0:j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))
        
        theta = temp
        cost[i] = ComputeCost(x, y, theta)
    return theta , cost

alpha = 0.01    
iters = 1000

g , cost = GradiantDescient(x,y,theta,alpha,iters)

print('g : \n' , g)
print('cost : \n' , cost[-1])

