#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[20]:


data = pd.read_csv('Glucose Level - T1D - FreeStyle Libre.csv')
data.head()


# In[21]:


data = data.to_numpy()


# In[22]:


data.shape


# In[23]:


lista_usuarios = np.unique(data[:, 0])
lista_usuarios


# In[24]:


usuarios_data = {}
usuarios_data_descartados = {}
umbral = 10000
for usuario in lista_usuarios:
    cadena = data[np.where(data[:, 0] == usuario)[0], 1:]
    if len(cadena) > umbral:
        usuarios_data[usuario] = cadena
    else:
        usuarios_data_descartados[usuario] = cadena
        print(" > Se descarta el usuario", usuario)


# In[25]:


np.min([len(usuarios_data[usuario]) for usuario in usuarios_data.keys()])


# In[26]:


len(usuarios_data.keys())


# In[27]:


usuarios_data['LIB193263']


# In[28]:


plt.hist([len(usuarios_data[usuario]) for usuario in usuarios_data.keys()])
plt.show()


# In[29]:


for usuario in usuarios_data.keys():
    usuarios_data[usuario][:, 0] = np.array([float(fecha.split()[1][:2]) for fecha in usuarios_data[usuario][:, 0]], dtype=np.float64)
    
for usuario in usuarios_data_descartados.keys():
    usuarios_data_descartados[usuario][:, 0] = np.array([float(fecha.split()[1][:2]) for fecha in usuarios_data_descartados[usuario][:, 0]], dtype=np.float64)


# In[30]:


usuarios_data['LIB193263']


# In[31]:


tamano_secuencia_maximo = 10000

secuencias = []
for usuario in usuarios_data.keys():
    num_cadenas = len(usuarios_data[usuario]) // tamano_secuencia_maximo
    for i in range(num_cadenas - 1):
        secuencias.append(usuarios_data[usuario][i*tamano_secuencia_maximo:(i+1)*tamano_secuencia_maximo])
secuencias = np.array(secuencias)

np.save('dataset_training.npy', secuencias)


# In[32]:


tamano_secuencia_maximo = 10000

secuencias = []
for usuario in usuarios_data_descartados.keys():
    secuencias.append(usuarios_data_descartados[usuario])
secuencias = np.array(secuencias, dtype=object)

np.save('dataset_test.npy', secuencias)

