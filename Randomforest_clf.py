#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')


# In[3]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[4]:


print(mnist.data)
print(mnist.data.shape)


# In[5]:


mnist.data[0]


# In[6]:


first_image = mnist.data[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()


# In[7]:


print(mnist.target)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7.0)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[10]:


y_pred = model.predict(X_test)


# In[11]:


## RF classifier score
model.score(X_test, y_test)


# In[12]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_test)


# In[13]:


n=1
X_test[n]


# In[14]:


y_test[n]


# In[15]:


y_predicted = model.predict(X_test[n].reshape(1,-1))
f_image = X_test[n]
f_image = np.array(f_image, dtype='float')
pixels = f_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.title('Label is {label}'.format(label=y_predicted))
plt.show()


# In[22]:


model.set_params


# In[ ]:




