from flask import Flask, render_template, url_for, Response
import base64
from datetime import datetime
import pandas as pd
import io
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from forecasting import SunForecast
from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)

sun_forecast = SunForecast()


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/mpei')
def mpei():
    tz = 'Europe/Moscow'
    lat, lon = 55.75497, 37.709
    times = pd.date_range('2020-01-01 00:00:00', '2021-01-01', closed='left',
                          freq='H', tz=tz)

    solpos = solarposition.get_solarposition(times, lat, lon)
    # remove nighttime
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    fig, ax = plt.subplots(figsize=(18, 8))

    points = ax.scatter(solpos.azimuth, solpos.apparent_elevation, s=2,
                        c=solpos.index.dayofyear, label=None)
    # fig.colorbar(points)

    for hour in np.unique(solpos.index.hour):
        # choose label position by the largest elevation for each hour
        subset = solpos.loc[solpos.index.hour == hour, :]
        height = subset.apparent_elevation
        pos = solpos.loc[height.idxmax(), :]
        ax.text(pos['azimuth'], pos['apparent_elevation'], str(hour))

    for date in pd.to_datetime(['2019-03-21', '2019-06-21', '2019-12-21', datetime.now().strftime("%d-%m-%y")]):
        times = pd.date_range(date, date + pd.Timedelta('24h'), freq='5min', tz=tz)
        solpos = solarposition.get_solarposition(times, lat, lon)
        solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
        label = date.strftime('%Y-%m-%d')
        ax.plot(solpos.azimuth, solpos.apparent_elevation, label=label)

    ax.scatter(solarposition.get_solarposition(datetime.utcnow(), lat, lon).azimuth,
               solarposition.get_solarposition(datetime.utcnow(), lat, lon).apparent_elevation, s=1200, label='now')
    im = plt.imread("mpei_pano2.jpg")
    implot = plt.imshow(im, origin='upper', extent=[0, 360, 0, 25])

    # plt.scatter([100,8000], [0, 1500])

    # ax.plot(x, x, '--', linewidth=5, color='firebrick')
    ax.figure.legend(loc='upper left')
    ax.set_xlabel('Solar Azimuth (degrees)')
    ax.set_ylabel('Solar Elevation (degrees)')

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    #<!><center><a href="https://ibb.co/BBVk3vx"><img src="https://i.ibb.co/rHG90qS/PANO-20211008-124808-2.jpg" alt="PANO-20211008-124808-2" border="1"></a></center><-->

    return render_template('mpei.html', image=pngImageB64String)


@app.route('/user/<string:name>/<int:id>')
def user (name, id):
    return "User page: " + name + " - " + str(id)


@app.route("/nvchb", methods=["GET"])
def plotView():
    # Generate plot
    plt.style.use('seaborn-white')
    fig = Figure(figsize=(8, 5))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Прогноз сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"))
    axis.set_xlabel("Дата и время")
    axis.set_ylabel("Среднечасовая интенсивность солнечной \nрадиации, Вт/кв.м")
    forecast_itself = sun_forecast.get_forecast()
    axis.plot(forecast_itself)
    axis.xaxis_date()
    axis.grid()
    #axis.plot(range(5), range(5), "ro-")

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return render_template("nvchb.html", image=pngImageB64String,
                           tables=[pd.DataFrame(forecast_itself).to_html()],
                           titles=pd.DataFrame(forecast_itself).columns.values)


if __name__ == "__main__":
    app.run(debug=False)