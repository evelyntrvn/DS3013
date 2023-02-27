import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor

# print(matplotlib.get_backend())

df = pd.read_csv('Number of Victims Female cleaned.csv')
df2 = pd.read_csv('Number of Victims Male cleaned.csv')

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
    y_train = commercial[::2]
    x_test = times[1::2]
    x_test1 = np.array(mappedArray(timesToNumMap, x_test)).reshape(-1, 1)
    y_test = commercial[1::2]

    # print(x_test)
    print(x_train)
    print(y_train)

    # linear regression
    model = linear_model.LinearRegression()
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


visualize(df, "Female")
visualize(df2, "Male")



