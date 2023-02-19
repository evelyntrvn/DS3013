import pandas as pd

df1 = pd.read_csv('Number of Victims by Age by Gender by County.csv', sep=";", encoding='cp1252')
print(df1.head())

