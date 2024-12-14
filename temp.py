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
print ('XXXXX \n',x)
y = data.iloc[ : ,cols -1 : cols]
print ('YYYYYY \n',y)

# print ('x data = \n',x.head(10))
# print ('*' * 40)
# print ('y data = \n',y.head(10))
# print ('*' * 40)
#
#convert data from data frame to numpy matrix
x = np.matrix(x.values) # convert x 
y = np.matrix(y.values) # convert y 
theta = np.zeros((1,x.shape[1]))
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
    # create temp to use the same theta for all process 
    #   if i use theta and make override theta will change 
    #   and this is wrong
    temp = np.zeros(theta.shape) 
    parameters = theta.shape[1] # number of columns theta matrix
    cost = np.zeros(iters) # matrix zeroes (iters : number of columns)
    for i in range(iters):
        error = (x * theta.T) - y
        for j in range(parameters): # 0 , 1
            term = np.multiply(error,x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x)) * np.sum(term))       
        theta = temp
        cost[i] = ComputeCost(x, y, theta)
    return theta , cost

alpha = 0.01    
iters = 1000

g , cost = GradiantDescient(x,y,theta,alpha,iters)

print('g : \n' , g)
print('compute cost : \n' , cost[-1])

# get best fit line 
# linspace :divide range between min and max to 100 parts
x_line = np.linspace(data.Population.min(),data.Population.max(),100) 
# print ('X\n',x[99]) 
# print ('g\n',g.shape)

f = g[0,0] + (g[0,1] * x_line)

# draw best fit line
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(x_line,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc = 2)
ax.set_xlabel("Population")
ax.set_ylabel("Profit")
ax.set_title('Predicted Profit vs. Populatin Size')
plt.show()

# Draw the error graph
fig, ax = plt.subplots(figsize=(5, 5))  # Create a figure and axis with a 5x5 size
ax.plot(np.arange(iters), cost, 'r', label="Cost")  # Plot cost over iterations
ax.set_xlabel('Iterations')  # Label for x-axis
ax.set_ylabel('Cost')  # Label for y-axis
ax.set_title('Error vs. Training Epoch')  # Title of the plot
ax.legend()  # Add a legend to indicate what the line represents
plt.grid(True)  # Add grid lines for better readability
plt.show()

