import math

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from matplotlib.figure import Figure
import numpy as np
from numpy import concatenate
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor

# print(matplotlib.get_backend())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('Number of Victims Female cleaned.csv')
df2 = pd.read_csv('Number of Victims Male cleaned.csv')
n = 'dataset.csv'

def mapping(array):
    map = {}

    for i in range(len(array)):
        map[array[i]] = i

    # print(map)
    return map


def mappedArray(mapping, array):
    result = []

    for i in range(len(array)):
        result.append(mapping[array[i]])

    # print(result)
    return result

def visualize(datafile, gender):
    plt.Figure(figsize=(20, 20))
    df1 = pd.DataFrame(datafile)

    df2 = df1.transpose()
    columns = ['Commercial', 'Government/Public Building and other', 'Road/Parking/Camps']
    df2.columns = columns
    df2.drop(index=df2.index[0], axis=0, inplace=True)
    times = df2.index.to_numpy()
    commercial = df2['Commercial'].to_numpy()
    government = df2['Government/Public Building and other'].to_numpy()
    road = df2['Road/Parking/Camps'].to_numpy()

    # dividing up data set
    timesToNumMap = mapping(times)  # mapping of times to number

    x_train = times[::2]
    x_train1 = np.array(mappedArray(timesToNumMap, x_train)).reshape(-1, 1)
    y_train = commercial[::2].astype('int')
    x_test = times[1::2]
    x_test1 = np.array(mappedArray(timesToNumMap, x_test)).reshape(-1, 1)
    y_test = commercial[1::2]

    # print(x_test)
    print(x_train)
    print(y_train)

    # linear regression
    model = linear_model.LogisticRegression()
    model.fit(x_train1, y_train)
    y_pred = model.predict(x_test1)
    # print(y_pred)
    # print(y_test)

    plt.plot(x_test1, y_pred, label='Commercial')
    plt.scatter(x_train, y_train)
    # plt.plot(times, government, label='Government/Public Building and other')
    # plt.plot(times, road, label='Road/Parking/Camps')

    # Add labels and legend
    plt.xlabel('Time of day')
    xticks = [0, 1, 2, 3, 4, 5, 6, 7]
    plt.xticks(xticks, times)
    plt.ylabel('Number of incidents')
    plt.title('Incidents by time of day and location type ' + gender)
    plt.legend()

    # Display plot
    plt.show()

def lstm(fileName):
    datafile = pd.read_csv(fileName)
    df1 = pd.DataFrame(datafile)
    df1 = df1.fillna(0)

    df1 = df1.drop(columns="Offense Type")
    df1 = df1.drop(columns="Incident Date")

    for i in df1.select_dtypes('object').columns:
        le = LabelEncoder().fit(df1[i])
        df1[i] = le.transform(df1[i])

    sc = MinMaxScaler()
    sc = sc.fit(df1[['Location Type', 'Victim Gender', 'Incident Hour of Day', 'Number of Victims']])
    X = sc.transform(df1[['Location Type', 'Victim Gender', 'Incident Hour of Day', 'Number of Victims']])

    sc2 = MinMaxScaler()
    sc2 = sc2.fit(df1[['Number of Victims']])
    y = sc2.transform(df1[['Number of Victims']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')

    history = model.fit(X_train, y_train, epochs=20, batch_size=72, validation_data=(X_test, y_test), verbose=2,
                        shuffle=False)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'validation loss'])
    plt.show()

    print(X_test[:1])

    # make a prediction
    yhat = model.predict(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
    inv_yhat = sc.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    y_test = np.array(y_test)
    y_test = y_test.reshape((len(y_test), 1))
    inv_y = concatenate((y_test, X_test[:, 1:]), axis=1)
    inv_y = sc.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    plt.plot(inv_y, label='actual', color='#ff000080')
    plt.plot(inv_yhat, label='predicted', color='#0000ff80')
    plt.ylabel('Number of Victims')
    # plt.xlabel('Days (2018-2021)')
    plt.legend()
    plt.show()

    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


lstm(n)

visualize(df, "Female")
visualize(df2, "Male")



