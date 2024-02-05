#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().system('pip install tensorflow')


# In[77]:


from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer ()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


data


# In[79]:


data.data


# In[80]:


data.target


# In[81]:


x = pd.DataFrame(data.data)
x


# In[82]:


y = pd.DataFrame(data.target)
y


# # TestTrain 

# In[83]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=30,random_state=43)


# In[84]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # TF learn 

# In[85]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.optimizers import Adam


# In[86]:


# model initialisation
model = Sequential()
model.add(Input(30,))
model.add(Dense(units=32,activation = 'relu'))
model.add(Dense(units=32,activation = 'relu'))
model.add(Dense(units=1,activation='softmax'))


# In[87]:


#Model Compile
model.compile(optimizer='rmsprop',loss='mse',metrics='accuracy')


# In[88]:


model.fit(x=x_train,y=y_train,epochs=100,validation_data=(x_test,y_test))


# In[72]:


model.history.history.keys()


# In[75]:


train_loss= model.history.history["loss"]
val_loss = model.history.history["val_loss"]


# In[89]:


train_loss= model.history.history["loss"]
val_loss = model.history.history["val_loss"]

plt.train(train_loss)
plt.plot(val_loss)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(["train","test"])
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




