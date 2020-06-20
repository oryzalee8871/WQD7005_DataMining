import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import statsmodels.api as sm
import streamlit as st


import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from Time_Series_Prediction_Palm_oil_with_func import Time_Series



st.title("Prediction Model for Crude Palm Oil Price")
st.markdown(
"""
This is a dashboard showing result of different prediction model
to predict crude palm oil price using 
LSTM and ARIMA

"""
)



dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
# os.chdir('C:\\Users\\oryza\\OneDrive\\Desktop\\DataScience\\DataMiningOnClass\\project')
CPO_Price=pd.read_csv("C:\\Users\\oryza\\OneDrive\\Desktop\\DataScience\\WQD7005_DataMining\\Source Code\\04_1_StreamLit_WebApp\\investing_Bursa_CPO_USD_price.csv",parse_dates=['Date'], date_parser=dateparse)

CPO_Price_September=CPO_Price[CPO_Price['Date']>='2014-09-01']
CPO_Price_September=CPO_Price_September.drop(['Open','High','Low','Vol.','Change %'],axis=1)
CPO_Price_September.set_index('Date',inplace=True)

'Crude palm oil price dataset from  Sep 2014 to March 2020',CPO_Price_September


Type_Of_Prediction_Model=st.selectbox("Choose Your Prediction Model?    (PS: Longer run time for LSTM)", 
                         ["-","LSTM","ARIMA"])



if Type_Of_Prediction_Model=="LSTM":
    with st.spinner('Please wait for it,algorithm is running, this can take up to 10 mins'):
        time.sleep(5)
        Time_Series()
    st.success('Done!')





elif Type_Of_Prediction_Model=="ARIMA":

    #Test for staionarity
    def test_stationarity(timeseries):
        #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        #Plot rolling statistics:
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        # plt.show(block=False)
        st.pyplot()

        
        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        # output for dft will give us without defining what the values are.
        #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        print(output)
        
    # test_stationarity(CPO_Price_September)
    CPO_Price_September_edit=CPO_Price_September.iloc[::-1]


    result = seasonal_decompose(CPO_Price_September_edit, model='multiplicative', freq = 30)
    # fig = plt.figure()  
    # fig = result.plot() 
    # st.pyplot(fig)
    # fig.set_size_inches(16, 9)



    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 6
    df_log = np.log(CPO_Price_September_edit)
    moving_avg = df_log.rolling(5).mean()
    std_dev = df_log.rolling(5).std()
    plt.legend(loc='best')
    plt.title('Moving Average')
    plt.plot(std_dev, color ="black", label = "Standard Deviation")
    plt.plot(moving_avg, color="red", label = "Mean")
    plt.legend()



    #split data into train and training set
    train_data, test_data = df_log[3:int(len(df_log)*0.94)], df_log[int(len(df_log)*0.94):]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')



    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
    test='adf',       # use adftest to find             optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1,              # frequency of series
                        d=None,           # let model determine 'd'
                        seasonal=True,   # No Seasonality
                        start_P=0, 
                        D=0, 
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

    model = ARIMA(train_data, order=(1, 2, 5))  
    # model=model_autoARIMA
    fitted = model.fit(disp=-1)  


    # Forecast
    fc, se, conf = fitted.forecast(82, alpha=0.05)  # 95% confidence
    # fc, se, conf = model_autoARIMA.forecast(88, alpha=0.15)  # 95% confidence
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train_data, label='training')
    plt.plot(test_data, color = 'blue', label='Actual CPO Price')
    plt.plot(fc_series, color = 'orange',label='Predicted CPO Price')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.10)
    plt.title('Crude Palm Oil Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Scale CPO Price')
    plt.legend(loc='upper left', fontsize=8)
    st.pyplot()

    mse = mean_squared_error(test_data, fc)
    'Mean Squared Error: ',str(mse)
 
    rmse = math.sqrt(mean_squared_error(test_data, fc))
    'Root Mean Square Error: ',str(rmse)




 



