import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import statsmodels.api as sm
import streamlit as st


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import math
from sklearn.preprocessing import MinMaxScaler
plt.style.use('fivethirtyeight')

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
os.chdir('C:\\Users\\oryza\\OneDrive\\Desktop\\DataScience\\DataMiningOnClass\\project')
CPO_Price=pd.read_csv("..\\WQD7005_DataMining\\A_Raw_Data\\CPOPrices\\investing_Bursa_CPO_USD_price.csv",parse_dates=['Date'], date_parser=dateparse)

CPO_Price_September=CPO_Price[CPO_Price['Date']>='2014-09-01']
CPO_Price_September=CPO_Price_September.drop(['Open','High','Low','Vol.','Change %'],axis=1)
CPO_Price_September.set_index('Date',inplace=True)

def Time_Series():
    y = CPO_Price_September['Price'].resample('SMS').mean()


    y.plot(figsize=(15, 6))
    # st.pyplot()

    # plt.show()

    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',freq=40)
    fig = decomposition.plot()
    # st.pyplot()
    # plt.show()pip


    # normalize the data_set 
    sc = MinMaxScaler(feature_range = (0, 1))
    df = sc.fit_transform(CPO_Price_September)

    # split into train and test sets
    train_size = int(len(df) * 0.75)
    test_size = len(df) - train_size
    train, test = df[0:train_size, :], df[train_size:len(df), :]


    # convert an array of values into a data_set matrix def
    def create_data_set(_data_set, _look_back=1):
        data_x, data_y = [], []
        for i in range(len(_data_set) - _look_back - 1):
            a = _data_set[i:(i + _look_back), 0]
            data_x.append(a)
            data_y.append(_data_set[i + _look_back, 0])
        return np.array(data_x), np.array(data_y)

    # reshape into X=t and Y=t+1
    look_back =90
    X_train,Y_train,X_test,Ytest = [],[],[],[]
    X_train,Y_train=create_data_set(train,look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test,Y_test=create_data_set(test,look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    # create and fit the LSTM network regressor = Sequential() 
    regressor = Sequential()

    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.1))

    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.1))

    regressor.add(Dense(units = 1))


    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5)
    history =regressor.fit(X_train, Y_train, epochs = 20, batch_size = 15,validation_data=(X_test, Y_test), callbacks=[reduce_lr],shuffle=False)



    train_predict = regressor.predict(X_train)
    test_predict = regressor.predict(X_test)

    # invert predictions
    train_predict = sc.inverse_transform(train_predict)
    Y_train = sc.inverse_transform([Y_train])
    test_predict = sc.inverse_transform(test_predict)
    Y_test = sc.inverse_transform([Y_test])

    # print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
    # print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
    # print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
    # print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    # st.pyplot()

    # plt.show()



    #Compare Actual vs. Prediction
    aa=[x for x in range(180)]
    plt.figure(figsize=(8,4))
    plt.plot(aa, Y_test[0][:180], marker='.', label="actual")
    plt.plot(aa, test_predict[:,0][:180], 'r', label="prediction")
    plt.tight_layout()
    sns.despine(top=True)
    plt.subplots_adjust(left=0.07)
    plt.ylabel('Price', size=15)
    plt.xlabel('Time step', size=15)
    plt.legend(fontsize=15)
    st.pyplot()


    # plt.show()
    # Print Parameters
    
   
    st.write('Mean Squared Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
    st.write('Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))

