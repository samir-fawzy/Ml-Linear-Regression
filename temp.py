import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
path = 'C:\\New folder (2)\\Python\\data.txt'
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
# 
