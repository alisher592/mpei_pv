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
from flask_sqlalchemy import SQLAlchemy
import urllib
import bs4
import requests
import json
from flask_wtf import Form
#from wtforms.fields.html5 import DateField
from flask_bootstrap import Bootstrap
from flask_datepicker import datepicker



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
Bootstrap(app)
db = SQLAlchemy(app)
app.debug = True

ntc_url = 'http://ntc.nudl.net/show_nchb.php'

# class Row(db.Model):
#     __tablename__ = "data_history"
#     #id = db.Column(db.Integer, primary_key=True)
#     datetime = db.Column(db.DateTime, default=datetime.utcnow())
#     datetime_idx = db.Column(db.String, primary_key=True)
#     poa_fcst = db.Column(db.Float)
#
#     def __repr__(self):
#         return '<Row %r>' % self.id




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

    fig, ax = plt.subplots(figsize=(12, 4))

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
    t = 0
    if t == 0:
        im = plt.imread("mpei_pano2_cmpr.jpg")

    implot = plt.imshow(im, origin='upper', extent=[0, 360, 0, 21])

    # plt.scatter([100,8000], [0, 1500])

    # ax.plot(x, x, '--', linewidth=5, color='firebrick')
    # ax.figure.legend(loc='upper left')
    ax.set_xlabel('Solar Azimuth (degrees)')
    ax.set_ylabel('Solar Elevation (degrees)')
    #ax.set_aspect('auto')
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

    fig2 = Figure(figsize=(8, 5))
    axis2 = fig2.add_subplot(1, 1, 1)
    axis2.set_title("Запрос данных мониторинга сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"))
    axis2.set_xlabel("Дата и время")
    axis2.set_ylabel("Интенсивность солнечной \nрадиации, Вт/кв.м")

    forecast_itself = sun_forecast.get_forecast()

    ntc_data = ntc_data_loader()
    print(ntc_data)
    #print(forecast_itself.index.strftime("%Y/%m/%d %H:%M"))




    # row=Row()
    #
    # for r in range(0, len(forecast_itself)):
    #     row.poa_fcst = forecast_itself[r]
    #     row.datetime_idx = forecast_itself.index[r].strftime("%Y-%m-%d %H:%M")
    #     db.session.add(row)
    #
    # db.session.commit()

    #row = Row(poa_fcst=forecast_itself.values, datetime_idx=np.array(forecast_itself.index.strftime("%Y/%m/%d %H:%M")))

    #try:
    #db.session.add(row)
    #db.session.commit()

    #except:
        #return "Ошибка!"

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

    axis2.plot(forecast_itself, label = 'Прогноз')
    axis2.plot(ntc_data[0], label = "Факт. данные \n в исходном разрешении")
    axis2.plot(ntc_data[1], label="Факт. данные \n с часовым усреднением")
    axis2.xaxis_date()
    axis2.grid()
    axis2.legend(loc='best')
    # axis.plot(range(5), range(5), "ro-")

    # Convert plot to PNG image
    pngImage2 = io.BytesIO()
    FigureCanvas(fig2).print_png(pngImage2)

    # Encode PNG image to base64 string
    pngImageB64String2 = "data:image/png;base64,"
    pngImageB64String2 += base64.b64encode(pngImage2.getvalue()).decode('utf8')

    return render_template("nvchb.html", image=pngImageB64String, image2=pngImageB64String2,
                           tables=[pd.DataFrame(forecast_itself).to_html()],
                           titles=pd.DataFrame(forecast_itself).columns.values)


def ntc_data_loader(given_date=datetime.now().strftime("%Y-%m-%d")):
    target_date = datetime.strptime(given_date, '%Y-%m-%d').date()
    data = {'language': 'Русский', 'date_sel': target_date, 'ServerId': '100'}
    #print(urllib.request.urlopen(ntc_url).getcode(), type(urllib.request.urlopen(ntc_url).getcode()))
    if urllib.request.urlopen(ntc_url).getcode() == 200:
        soup = bs4.BeautifulSoup(requests.post(ntc_url, data).text, 'lxml')
        js = soup.find_all('script', {'type': 'text/javascript', 'language': 'javascript'})

        if len(js) != 0:
            Irradiance = js[0].string.split(';')[22][
                         21:len(js[0].string.split(';')[22])]  # комментарий от 2.12.20. Пришлось
            # заменить метод для .text BeautifulSoupTag  на .string из-за изменений в библиотеке BeautifulSoup
            result = json.loads(Irradiance)
            I_raw = np.array(result)  # сырой массив с мгновенными данными по СР
            Irr = pd.DataFrame(I_raw)
            Irr[0] = pd.to_datetime(Irr[0], unit='ms')  # преобразование
            Irr = Irr.set_index(0)
            Irr = Irr.tz_localize('UTC').tz_convert('Europe/Moscow')  # подстройка под часовой пояс Москвы
            Irr = Irr.tz_localize(None)
            Irr_mean = Irr.groupby(Irr.index.to_period('H')).mean()  # среднечасовые значения СР
            Irr_mean.index = Irr_mean.index.to_timestamp()
        else:
            Irr, Irr_mean = None, None
    else:
        Irr, Irr_mean = None, None
    return Irr, Irr_mean




if __name__ == "__main__":
    app.run(debug=True)