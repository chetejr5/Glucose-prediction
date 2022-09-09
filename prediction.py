"Importamos las librerías"
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import seaborn as sns

import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import numpy
import matplotlib.pyplot as plt
import pandas
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalMaxPool1D
from keras.layers import LSTM
from keras import Input


'''
from sktime.classification.interval_based import TimeSeriesForestClassifier

'''

from sklearn.metrics import accuracy_score

"Datos del Excel importados"
data = pd.read_csv('TFM_RECORTADO.csv')
data.head()


"Se procede a hacer un procesado de los datos"
"Lo primero que haremos será rellenar los huecos en blanco, si los hubiera, propagando el último valor válido"

data = data.iloc[1:]
data = data.fillna(method='ffill')
data.head()
print(data)


'''
"Mostramos los diferentes ID que tenemos en la misma gráfica"
data.Date = pd.to_datetime(data.Date)
sns.lineplot(data=data, x="Date", y="Glucose level", hue="ID")
plt.xticks(rotation=90)
'plt.show()'

"Ahora graficamos en gráficas separadas"
sns.relplot(data=data, kind="line",x="Date", y="Glucose level", row="ID", height=2, aspect=3)
plt.xticks(rotation=90)
'plt.show()'

'''

"Analizamos los diferentes ID que hemos de estudiar"
ID = data['ID'].unique()
#print(ID)

"Dividimos los datos en X e Y"
data_x = data['Glucose level'][:-1]
data_y = data['Glucose level'][1:]




"Dividimos entre datos de training y datos de test"

x_train = np.copy(data_x[:3800])
y_train = np.copy(data_y[0:3800])

x_test = np.copy(data_x[3800:])
y_test = np.copy(data_y[3800:])

"Normalizamos los datos"

media = np.mean(x_train)
desviacion = np.std(x_train)
mediay = np.mean(y_train)
desviaciony = np.std(y_train)

x_train = (x_train - media)/desviacion
x_test = (x_test - media)/desviacion
y_train = (y_train - mediay)/desviaciony
y_test = (y_test - mediay)/desviaciony

max_length_sentence = 20
batch_size = 20
LSTM_units = 50


def reshape_for_rnn(data, batch_size, max_length_sentence):
    data_size = len(data) // batch_size // max_length_sentence * batch_size * max_length_sentence
    data = data[:data_size]

    num_batches = data_size // batch_size

    data_temp = data.reshape(batch_size, num_batches).T
    data_ready = data_temp.reshape(num_batches // max_length_sentence,
                                   max_length_sentence, batch_size)
    final_data = np.concatenate(data_ready, axis=1).T

    return final_data


x_train_final = reshape_for_rnn(x_train, batch_size, max_length_sentence)
x_test_final = reshape_for_rnn(x_test, batch_size, max_length_sentence)
y_train_final = reshape_for_rnn(y_train, batch_size, max_length_sentence)
y_test_final = reshape_for_rnn(y_test, batch_size, max_length_sentence)



def builtmodel (batch_size, max_length_sentence, LSTM_units):

    model = Sequential()
    model.add(Input(batch_shape=(batch_size, max_length_sentence,1)))
    model.add(LSTM(130, return_sequences=True,stateful=True, dropout=0.055))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.26))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam (learning_rate=0.006, clipvalue=64.20, decay=2.14e-5), loss='mean_squared_error')
    model.summary()
    return model

model = builtmodel(batch_size, max_length_sentence, LSTM_units)

min_loss = 9999999.0

for i in range(10):
    model.reset_states()
    entrenando = model.fit(x_train_final[:,:,None],y_train_final[:,:,None],
                       epochs=1, batch_size=batch_size, shuffle = False, validation_data = (x_test_final[:,:,None],y_test_final[:,:,None]) )
    if entrenando.history["val_loss"][0] < min_loss:
        min_loss = entrenando.history["val_loss"][0]
        model.save_weights("best")

#model.load_weights("best")

def predecir (model, x_train, n):
    model_predict = builtmodel(batch_size=1, max_length_sentence=1, LSTM_units=130)
    model_predict.set_weights(model.get_weights())
    model_predict.reset_states()
    x = model_predict.predict(x_train, steps =len(x_train))[-1:]
    preds=[x[0,0,0]]
    for i in range(n-1):
        x = model_predict.predict(x)
        preds.append(x[0,0,0])
    return np.array(preds)

preds = predecir(model, x_train.reshape(-1,1)[:,:,None], 20)
print(preds.shape)

preds = preds * desviacion + media
x_test = x_test *desviacion + media


plt.plot(x_test[0:20], label = "esperado")
plt.plot(preds[0:20], label = "real")
plt.legend()
plt.show()

print(x_test[0:20])
print(preds[0:20])

diff=np.subtract(x_test[0:20],preds)
square=np.square(diff)
MSE=square.mean()
RMSE=np.sqrt(MSE)
print("Root Mean Square Error:", RMSE)

print(np.corrcoef(x_test[0:20], preds))





def smape (a,f):
    return 1/len(a) * np.sum(2* np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

print(smape(x_test[0:20], preds))
