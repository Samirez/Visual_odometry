import csv

import numpy as np
import utm
import pandas as pd

# ---------------- reading the data ------------------------------------------------
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
location = np.array(len(longitude))
# location = utm.from_latlon(latitude[2], longitude[2])
'''for i in range(1, len(latitude)):
    location[i] = utm.from_latlon(latitude[i], longitude[i])

print(location)'''
