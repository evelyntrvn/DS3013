import pandas as pd
import matplotlib.pyplot as plt

dfMale= pd.read_csv('Number of Victims Female.csv', sep=";", encoding='cp1252')
dfFemale = pd.read_csv('Number of Victims Male.csv', sep=";", encoding='cp1252')

times = ['6pm-9pm', '9pm-12am', '12am-3am', '3am-6am', '6am-9am', '9am-12pm', '12pm-3pm', '3pm-6pm']
commercial = [192, 121, 195, 32, 45, 146, 180, 178]
government = [52, 39, 18, 10, 25, 36, 41, 39]
road = [333, 276, 342, 88, 77, 201, 295, 368]

# Create plot
plt.plot(times, commercial, label='Commercial')
plt.plot(times, government, label='Government/Public Building and other')
plt.plot(times, road, label='Road/Parking/Camps')

# Add labels and legend
plt.xlabel('Time of day')
plt.ylabel('Number of incidents')
plt.title('Incidents by time of day and location type')
plt.legend()

# Display plot
plt.show()


