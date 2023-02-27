import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib.figure import Figure
import numpy as np
import seaborn as sb
import os

# print(matplotlib.get_backend())

df = pd.read_csv('Number of Victims Female cleaned.csv')
df2 = pd.read_csv('Number of Victims Male cleaned.csv')


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
    print(df2)
    print(times)
    print(commercial)

    plt.plot(times, commercial, label='Commercial')
    plt.plot(times, government, label='Government/Public Building and other')
    plt.plot(times, road, label='Road/Parking/Camps')

    # Add labels and legend
    plt.xlabel('Time of day')
    plt.ylabel('Number of incidents')
    plt.title('Incidents by time of day and location type ' + gender)
    plt.legend()

    # Display plot
    plt.show()

visualize(df, "Female")
visualize(df2, "Male")


