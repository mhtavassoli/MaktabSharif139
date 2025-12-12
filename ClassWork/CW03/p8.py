import numpy as np
import matplotlib.pyplot as plt
daysList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dayArray = np.array(daysList)
print(f"The days= {dayArray}")

temperatureArray=np.random.randint(15,35,7)
print(f"The temperatures= {temperatureArray}")

meanTemperatureArray=np.mean(temperatureArray)
maxTemperatureArray=np.max(temperatureArray)
minTemperatureArray=np.min(temperatureArray)
print(f"The max, min, mean of temperature array = {maxTemperatureArray}, {minTemperatureArray}, {meanTemperatureArray}")

plt.figure(figsize=(11,6))
plt.plot(dayArray,temperatureArray,marker='o',color='red',linestyle='-')
plt.xlim(0,10)
plt.xlabel('Days',fontweight='bold')
plt.xticks(range(0, 11))
plt.ylabel('Temperature',fontweight='bold')
plt.title('Temperature vs Days')
plt.show()
