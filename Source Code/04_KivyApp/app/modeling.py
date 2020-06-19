import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta


def feature_transform(df):
    df.reset_index(inplace=True)
    df['day_of_week'] = df['Date'].apply(lambda x: x.weekday())
    df['Price_MAV5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['Vol_MAV5'] = df['Volume'].rolling(5, min_periods=1).mean()
    # df['Change_MAV5'] = df['Change %'].rolling(5, min_periods=1).mean()
    df['Price_MAV10'] = df['Close'].rolling(10, min_periods=1).mean()
    df['Vol_MAV10'] = df['Volume'].rolling(10, min_periods=1).mean()
    df['Price_MACD'] = df['Price_MAV5'] - df['Price_MAV10'] 
    df['Vol_MACD'] = df['Vol_MAV5'] - df['Vol_MAV10'] 
    # df['Change_MAV10'] = df['Change %'].rolling(10, min_periods=1).mean()
    return df

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def weighted_average(pred, future_target):
    pred = [[a*0.5, b*0.25, c*0.1, d*0.075, e*0.075] for  a,b,c,d,e in pred]
    new_pred = []
    for i in range(len(pred)-future_target):
        hpad = [np.nan] * i if i > 0 else []
        tpad = [np.nan] * (len(pred)-future_target - i) if (len(pred)-future_target - i) > 0 else []
        hpad.extend(pred[i])
        hpad.extend(tpad)
        new_pred.append(hpad)
    new_pred = np.nansum(np.asarray(new_pred), axis=0)
    return new_pred


def plot_loss(df):
    loss = df['loss']
    val_loss = df['val_loss']
    
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(facecolor='#07000d')
    plt.plot(np.arange(len(loss)), loss ,label='train', linewidth=3)
    plt.plot(np.arange(len(loss)), val_loss ,label='validation', linewidth=3)
    plt.legend()
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('MAE')
    
    fig.canvas.draw_idle()
    return fig

def create_model(input_shape,n_output):
    tf.random.set_seed(123)
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape, recurrent_dropout=0.5, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(n_output, activation='linear'))

    model.compile(optimizer='nadam', loss='mae', metrics=['mae'])
    return model

def modeling(df, security_name):
    df = feature_transform(df)
    index = df['Date']
    df.drop('Date', axis=1, inplace=True)
    TRAIN_SPLIT= int(len(df)*(1-0.33))
    attributes=df.columns
    multi_data = df.loc[:,attributes]

    dataset = multi_data.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)

    # standardize data
    dataset = (dataset-data_mean)/data_std


    EPOCHS = 65
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    EVALUATION_INTERVAL = 15

    # past_history = 20
    # future_target = 1
    # STEP = 1
    # x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
    #                                                 TRAIN_SPLIT, past_history,
    #                                                 future_target, STEP,
    #                                                 single_step=True)
    # x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
    #                                             TRAIN_SPLIT, None, past_history,
    #                                             future_target, STEP,
    #                                             single_step=True)
                                                

    # train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
    # train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    # val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    # val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


    past_history = 20
    future_target = 5
    STEP = 1

    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target, STEP,
                                                    single_step=False)
    x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                                TRAIN_SPLIT, None, past_history,
                                                future_target, STEP,
                                                single_step=False)

    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    input_shape = x_train_multi.shape[-2:]
    n_output= future_target 
    model = create_model(input_shape, n_output)
    callback = tf.keras.callbacks.EarlyStopping(patience=20)
    csv_logger = tf.keras.callbacks.CSVLogger('training.csv', separator=",", append=False)
    history = model.fit(train_data_multi, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data= val_data_multi,
                        validation_steps=100, verbose=0,
                        callbacks=[callback, csv_logger])

    val_mae = round(history.history['val_mae'][-1], 3)

    val_pred = model.predict(x_val_multi)
    # to compute weighted average later
    new_val_pred = weighted_average(val_pred, future_target)
    

    x_future_multi = []
    # for i in range(future_target):
    #     x_future_multi.append(dataset[-past_history-future_target+i:-future_target+i].tolist())
    x_future_multi.append(dataset[-past_history:].tolist())
    x_future_multi = np.array(x_future_multi)
    new_future_pred = model.predict(x_future_multi)[0]
    # new_future_pred = weighted_average(future_pred, future_target)
    future_y_pred_index = [index.iloc[-1]+timedelta(days=i) for i in range(future_target)]

    try:
        ### Validation chart ###
        plt.style.use('dark_background')
        plt.rcParams.update({'font.size': 15})
        fig_val = plt.figure(facecolor='#07000d')
        plt.plot(index, df['Close'], label='Actual', linewidth=3, alpha=0.5)
        y_pred = pd.Series(new_val_pred, index=index[TRAIN_SPLIT+past_history:-future_target]).rename('Predicted')
        y_pred = (y_pred*data_std[0])+data_mean[0]
        plt.plot(y_pred.index, y_pred, label='Predicted', linewidth=3, linestyle='--')
        
        plt.xticks(rotation=45)
        plt.ylabel('Close Price')
        plt.xlabel('Datetime')
        plt.legend()
        plt.title(f'Validation Set Comparison on {security_name.upper()}\nMAE = {val_mae}')

        fig_val.canvas.draw_idle()
        
        ### Future Predict Chart ###

        fig_future = plt.figure(facecolor='#07000d')
        plt.plot(index.iloc[-30:], df['Close'].iloc[-30:], label='Actual', linewidth=3, alpha=1)
        future_y_pred_index = [index.iloc[-1]+timedelta(days=i+1) for i in range(future_target)]
        future_y_pred = pd.Series(new_future_pred, index=future_y_pred_index).rename('Predicted')
        future_y_pred = (future_y_pred*data_std[0])+data_mean[0]
        print(future_y_pred_index, future_y_pred)
        plt.plot(future_y_pred_index, future_y_pred, label='_nolegend_', color='orange', linewidth=3, linestyle='--', alpha=0.5)
        plt.scatter(future_y_pred_index, future_y_pred, label='Future Time Steps', color='orange', marker='x')

        plt.ylabel('Close Price')
        plt.xlabel('Datetime')
        plt.legend()
        
        plt.title(f'Close Price Prediction on {security_name.upper()}\n5 future time steps (days)')
        fig_future.canvas.draw_idle()
        return fig_val, fig_future
    except Exception as e:
        print(e)
        fig = plt.figure(facecolor='#07000d')
        fig.canvas.draw_idle()
        return fig