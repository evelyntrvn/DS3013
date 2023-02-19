import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
import numpy as np
import seaborn as sb
import os

os.chdir("Users/maya/DS3013")

df = pd.read_csv('Number of Victims by Age by Gender by County.csv', sep=";", encoding='cp1252')
print(df.head())

def visualize():
    plt.Figure(figsize=(20, 20))
    plt.plot(range(df.shape[0]), 100)
    plt.xticks(range(0, df.shape[0], 500), df['Incident Hour of Day'].loc[::500], rotation=45)
    plt.xlabel('Time of Day', fontsize=18)
    plt.ylabel('Number of Crimes', fontsize=18)
    plt.show()

visualize()