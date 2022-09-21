#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, GaussianNoise, SimpleRNN
from keras.optimizers import Adam, Nadam, RMSprop
from keras import Input
from keras import backend as K

from sklearn.metrics import mean_squared_error


# In[2]:


dataset_training = np.load('dataset_training.npy', allow_pickle=True)


# In[3]:


dataset_training.shape


# In[4]:


x_train = dataset_training[:150, :-1].astype(np.float64)
y_train = dataset_training[:150, 1:].astype(np.float64)
x_test = dataset_training[150:, :-1].astype(np.float64)
y_test = dataset_training[150:, 1:].astype(np.float64)


# In[21]:


media = x_train[:, :, 1].mean(axis=1)
desviacion = x_train[:, :, 1].std(axis=1)

x_train[:, :, 1] = (x_train[:, :, 1] - media[:, None]) / desviacion[:, None]
y_train[:, :, 1] = (y_train[:, :, 1] - media[:, None]) / desviacion[:, None]
x_test[:, :, 1] = (x_test[:, :, 1] - media.mean()) / desviacion.mean()
y_test[:, :, 1] = (y_test[:, :, 1] - media.mean()) / desviacion.mean()


# In[6]:


def build_model(batch_size, max_length_sentence, LSTM_units, 
                lstm_dropout=0.0, output_dropout=0.0, noise=0.0,
                learning_rate=1e-3, clipvalue=10.0, decay=1e-8,
                optimizer='Adam'):

    model = Sequential()
    model.add(Input(batch_shape=(batch_size, max_length_sentence,2)))
    model.add(GaussianNoise(noise))
    model.add(LSTM(LSTM_units, return_sequences=True,stateful=True, dropout=lstm_dropout))
    model.add(Dropout(output_dropout))
    model.add(Dense(1, activation='linear', use_bias=False))
    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, clipvalue=clipvalue, decay=decay)
    
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# In[7]:


def my_predict_at(model, x, n):
    model_predict = build_model(1, 1, lstm_units)
    model_predict.set_weights(model.get_weights())
    model_predict.reset_states()
    
    ultimos_cuatro = x[-4:, 0, 0]
    if ultimos_cuatro[-1] != ultimos_cuatro[-2]:
        contador = 0
    elif ultimos_cuatro[-1] != ultimos_cuatro[-3]:
        contador = 1
    elif ultimos_cuatro[-1] != ultimos_cuatro[-4]:
        contador = 2
    else:
        contador = 3
    ultima_hora = x[-1, 0, 0]
        
    x = model_predict.predict(x, steps=len(x), batch_size=1)[-1:]
    
    preds=[x[0, 0, 0]]
    for i in range(n-1):
        contador += 1
        if contador % 4 == 0:
            contador = 0
            ultima_hora += 1
        if ultima_hora % 24 == 0:
            ultima_hora = 0
            
        x = np.concatenate((np.array([ultima_hora])[None, None, :], x), axis=-1)
        x = model_predict.predict(x, batch_size=1, verbose=0)
        preds.append(x[0, 0, 0])
    return np.array(preds)


# In[8]:


max_length_sentence = 10
batch_size = 150
noise = 0.05
lstm_units = 2000
dense_units = 20 # Not used
lstm_dropout = 0.0
output_dropout = 0.0
optimizer = 'Adam'
learning_rate = 3e-4
clipvalue = 20.0
decay = 1e-4


# In[9]:


model = build_model(batch_size, max_length_sentence, lstm_units, noise=noise,
                    lstm_dropout=lstm_dropout, output_dropout=output_dropout,
                    optimizer=optimizer, learning_rate=learning_rate,
                    clipvalue=clipvalue, decay=decay)

model_test = build_model(len(x_test), max_length_sentence, lstm_units)


# In[10]:


model.summary()


# In[11]:


all_losses_train = []
all_losses_test = []
for epoch in range(200):
    print("\n > Epoch:", epoch)
    
    # ROLL DE LOS DATOS (SHUFFLE DE SECUENCIAS)
    shift = np.random.randint(x_train.shape[1])
    x_train_final = np.roll(x_train, shift, axis=1)
    x_test_final = x_test
    y_train_final = np.roll(y_train, shift, axis=1)
    y_test_final = y_test
    
    model.reset_states()
    epoch_loss = []
    for i in range(0, max_length_sentence, 10000):
        loss = model.train_on_batch(x_train_final[:, i:i+max_length_sentence], 
                                    y_train_final[:, i:i+max_length_sentence, 1:]) # CUIDADO PORQUE y ES SOLO GLUCOSA
        epoch_loss.append(loss)
    all_losses_train.append(np.mean(epoch_loss))
    
    epoch_loss = []
    model_test.set_weights(model.get_weights())
    model_test.reset_states()
    for i in range(0, max_length_sentence, 10000):
        loss = model_test.test_on_batch(x_test_final[:, i:i+max_length_sentence],
                                        y_test_final[:, i:i+max_length_sentence, 1:])
        epoch_loss.append(loss)
    all_losses_test.append(np.mean(epoch_loss))
    
    plt.plot(all_losses_train, '.-', label="train")
    plt.plot(all_losses_test, '.-', label="test")
    plt.legend()
    plt.grid()
    plt.show()
        
    #entrenando = model.fit(x_train_final, y_train_final, 
    #                       batch_size=batch_size, 
    #                       shuffle = False,
                           #validation_data = (x_test_final, y_test_final))
    #if entrenando.history["val_loss"][0] < min_loss:
    #    min_loss = entrenando.history["val_loss"][0]
    #    model.save_weights("best")
    #    best_epoch = epoch
    model.save_weights("last")


# In[12]:


model.load_weights("last")

test_user = 0
input_length = 3000
prediction_steps = 52 # Intenta que sea divisible entre 4

preds_at_50 = my_predict_at(model, x_test[test_user][:input_length, None, :], prediction_steps)

print(" > MSE@50:", mean_squared_error(preds_at_50, x_test[test_user, input_length:input_length+prediction_steps, 1]))


# In[13]:


plt.figure(figsize=(20, 10))
plt.plot(preds_at_50, label="Prediccion")
plt.plot(x_test[test_user, input_length:input_length+prediction_steps, 1], label="Esperado")
plt.xticks(np.arange(prediction_steps), np.arange(0, prediction_steps//4).repeat(4) + x_test[test_user, input_length:][-1, 0], rotation='vertical')
plt.legend()
plt.show()


# In[14]:



preds_at_50  = preds_at_50* desviacion.mean() + media.mean()


# In[15]:


x_test[test_user, input_length:input_length+prediction_steps, 1] = x_test[test_user, input_length:input_length+prediction_steps, 1] * desviacion.mean() + media.mean()


# In[16]:


print(preds_at_50)
print(x_test[test_user, input_length:input_length+prediction_steps, 1])


# In[17]:


plt.figure(figsize=(20, 10))
plt.plot(preds_at_50, label="Prediccion")
plt.plot(x_test[test_user, input_length:input_length+prediction_steps, 1], label="Esperado")
plt.xticks(np.arange(prediction_steps), np.arange(0, prediction_steps//4).repeat(4) + x_test[test_user, input_length:][-1, 0], rotation='vertical')
plt.legend()
plt.show()


# In[18]:


diff=np.subtract(x_test[test_user, input_length:input_length+prediction_steps, 1][0:4],preds_at_50[0:4])
square=np.square(diff)
MSE=square.mean()
RMSE=np.sqrt(MSE)
print("Root Mean Square Error:", RMSE)

print(np.corrcoef(x_test[test_user, input_length:input_length+prediction_steps, 1][0:4], preds_at_50[0:4]))


# In[19]:


def smape (a,f):
    return 1/len(a) * np.sum(2* np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

print(smape(x_test[test_user, input_length:input_length+prediction_steps, 1], preds_at_50))


# In[20]:


print(np.corrcoef(x_test[test_user, input_length:input_length+prediction_steps, 1][0:10], preds_at_50[0:10]))


# In[ ]:




