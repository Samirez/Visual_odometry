import csv

import numpy as np
import utm
import pandas as pd

# ---------------- reading the data ------------------------------------------------
from matplotlib import pyplot as plt

data = []
with open('DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        data.append(lines)

print(data[0])
# RC_GPS.latitude, RC_GPS.longitude and there is also OSD.latitude and OSD.longitude
df = pd.DataFrame(data)
# print(df)
# loc = df['RC_GPS.longitude'] is empty use OSD
# --------------------- finding the gps coordinates ------------------------------------------------------
i = 0
long_ind = 0
lat_ind = 0
for x in data[0]:
    if x == 'OSD.longitude':
        long_ind = i
        print(i)
    elif x == 'OSD.latitude':
        lat_ind = i
        print(i)
    i += 1
# ------------------------ assigning columns to latitude and longtitude ----------------------------------
longitude = df.loc[1:, long_ind].astype(float)
latitude = df.loc[1:, lat_ind].astype(float)

# print(longitude)
# print(latitude[1])

# --------------------- converting to utm ----------------------------------------------------------------
# print(len(latitude))
# print(len(longitude))
location_list = []
for i in range(1, len(latitude)):
    location = 0
    location = utm.from_latlon(latitude[i], longitude[i])
    location_list.append(location)

print(location_list[1:4])
easting = []
northing = []
# updates the list for longitude and latitude for UTM format
for x in range(1, len(location_list)):
    for idx, val in enumerate(location_list[x]):
        print(idx, val)
        if idx == 0:
            easting.append(val)
        elif idx == 1:
            northing.append(val)

# starts from index 1
# visualizing the data from the given coordinates
plt.plot(easting, northing)
plt.show()


