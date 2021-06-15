#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM,Dense,Dropout
data=pd.read_csv("D:\\nlp\\trainset.csv")


# In[2]:


trainset = data.iloc[:,1:2].values
sc=MinMaxScaler(feature_range=(0,1))
train=sc.fit_transform(trainset)


# In[3]:


x_train=[]
y_train=[]
for i in range(60,len(train)-1):
    x_train.append(train[i-60:i,0])
    y_train.append(train[i,0])
x_train=np.array(x_train)
y_train=np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


# In[ ]:


a=Sequential()
a.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))
a.add(Dropout(0.2))
a.add(LSTM(units = 50,return_sequences = True))
a.add(Dropout(0.2))
a.add(LSTM(units = 50,return_sequences = True))
a.add(Dropout(0.2))
a.add(LSTM(units = 50))
a.add(Dropout(0.2))
a.add(Dense(units = 1))
a.compile(optimizer="adam",loss="mean_squared_error")
a.fit(x_train,y_train)
a.save("stock.h5")


# In[ ]:


p=a.predict(x_train)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(y_train,color = 'red', label = 'Real Price')
plt.plot(p, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[ ]:


y_t=y_train.reshape(-1,1)
y=sc.inverse_transform(y_t)
predict=sc.inverse_transform(p)


# In[ ]:


plt.plot(y,color = 'red', label = 'Real Price')
plt.plot(predict, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[ ]:




