from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

tz = 'Europe/Moscow'
lat, lon = 55.75497, 37.709
times = pd.date_range('2020-01-01 00:00:00', '2021-01-01', closed='left',
                      freq='H', tz=tz)

solpos = solarposition.get_solarposition(times, lat, lon)
# remove nighttime
solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

fig, ax = plt.subplots(figsize=(18,8))

points = ax.scatter(solpos.azimuth, solpos.apparent_elevation, s=2,
                    c=solpos.index.dayofyear, label=None)
#fig.colorbar(points)

for hour in np.unique(solpos.index.hour):
    # choose label position by the largest elevation for each hour
    subset = solpos.loc[solpos.index.hour == hour, :]
    height = subset.apparent_elevation
    pos = solpos.loc[height.idxmax(), :]
    ax.text(pos['azimuth'], pos['apparent_elevation'], str(hour))



for date in pd.to_datetime(['2019-03-21', '2019-06-21', '2019-12-21', datetime.now().strftime("%d-%m-%y")]):
    times = pd.date_range(date, date+pd.Timedelta('24h'), freq='5min', tz=tz)
    solpos = solarposition.get_solarposition(times, lat, lon)
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    label = date.strftime('%Y-%m-%d')
    ax.plot(solpos.azimuth, solpos.apparent_elevation, label=label)

ax.scatter(solarposition.get_solarposition(datetime.utcnow(), lat, lon).azimuth, solarposition.get_solarposition(datetime.utcnow(), lat, lon).apparent_elevation, s=1200, label = 'now')
im = plt.imread("mpei_pano2.jpg")
implot = plt.imshow(im, origin='upper', extent=[0, 360, 0, 25])



#plt.scatter([100,8000], [0, 1500])

#ax.plot(x, x, '--', linewidth=5, color='firebrick')
ax.figure.legend(loc='upper left')
ax.set_xlabel('Solar Azimuth (degrees)')
ax.set_ylabel('Solar Elevation (degrees)')

plt.show()