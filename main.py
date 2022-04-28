from flask import Flask, render_template, url_for, Response, request, flash, abort
import base64
#from datetime import datetime
#import pandas as pd
import io
from matplotlib.figure import Figure
#import matplotlib.pyplot as plt
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
from flask_mysqldb import MySQL
from sqlalchemy import create_engine
#from flask_wtf import Form
#from wtforms.fields.html5 import DateField
from flask_bootstrap import Bootstrap
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
#from flask_datepicker import datepicker
import pvlib
import MySQLdb
from boto.s3.connection import S3Connection
import os

#s3 = S3Connection(os.environ['DB_HOST'], os.environ['DB_USER'], os.environ['DB_PASSWORD'], os.environ['DB_SCHEMA'])

app = Flask(__name__)

Bootstrap(app)

app.config['MYSQL_HOST'] = '62.109.30.150'
app.config['MYSQL_USER'] = 'kovacevic'
app.config['MYSQL_PASSWORD'] = '4815162342'
app.config['MYSQL_DB'] = 'weather'

mysql = MySQL(app)

mydb = create_engine('mysql+pymysql://' + 'kovacevic' + ':' + '4815162342' + '@' + '62.109.30.150' + ':' + str(3306), echo=False)

app.debug = False



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
def index():
    return render_template('index.html', graphJSON=gm())

@app.route('/mg_modelling')
def mg_mdl():
    return render_template('mg_modelling.html')

@app.route('/build_forecasting_models')
def bld_fcst_mdl():



    return render_template('build_forecasting_models.html', graphJSON=gm())

@app.route('/custom_forecasting')
def custom_fcst():

    return render_template('custom_forecasting.html', graphJSON=gm())

def gm(country='United Kingdom'):
    df = pd.DataFrame(px.data.gapminder())

    fig = px.line(df[df['country'] == country], x="year", y="gdpPercap")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print(fig.data[0])
    # fig.data[0]['staticPlot']=True

    return graphJSON


@app.route('/callback', methods=['POST', 'GET'])
def cb():
    return gm(request.args.get('data'))


@app.route('/get_geo', methods=['POST', 'GET'])
def get_geo():
    lat = request.form['latitude']
    lon = request.form['longitude']
    alt = request.form['altitude']
    tz = request.form['timezone']

    location = pvlib.location.Location(lat, lon, tz='Europe/Moscow', altitude=alt)

    tmy = pvlib.iotools.get_pvgis_tmy(location.latitude, location.longitude, outputformat='epw', usehorizon=True,
                                      userhorizon=None, startyear=2005, endyear=2016,
                                      url='https://re.jrc.ec.europa.eu/api/', timeout=30)[0]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=(tmy.index.dayofyear - 1) * 24 + tmy.index.hour, y=tmy['temp_air'], name="Температура воздуха"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=(tmy.index.dayofyear - 1) * 24 + tmy.index.hour, y=tmy['ghi'], name="Суммарная солнечная радиация,<br>на горизонтальной поверхности"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=(tmy.index.dayofyear - 1) * 24 + tmy.index.hour, y=tmy['dhi'], name="Рассеянная солнечная радиация<br>на горизонтальной поверхности"),
        secondary_y=True,
    )

    # fig.add_trace(
    #    go.Scatter(x=tmy.index.dayofyear, y=tmy['dni'], name="Прямая солнечная радиация<br>на нормальной поверхности"),
    #    secondary_y=True,
    #)

    fig.update_layout(margin=dict(l=20, r=20),
        title_text="Типичный метеорологический год (TMY) по данным PVGIS", title_x=0.5,  legend=dict(
            orientation="h", xanchor="center", x=0.5, y=1.175))

    fig.update_xaxes(title_text="Порядковый номер часа TMY")
    fig.update_yaxes(title_text="<b>Температура воздуха</b>, °С", secondary_y=False)
    fig.update_yaxes(title_text="<b>Солнечная радиация</b> Вт/кв.м", secondary_y=True)


    #fig = px.line(tmy['temp_air'], x=tmy.index.dayofyear, y='temp_air')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    #json.dumps({'len': len(lat)})

    #print(tmy)

    return graphJSON





@app.route('/mpei_pv')
def mpei_pv():
    # tz = 'Europe/Moscow'
    # lat, lon = 55.75497, 37.709
    # times = pd.date_range('2020-01-01 00:00:00', '2021-01-01', closed='left',
    #                       freq='H', tz=tz)
    #
    # solpos = solarposition.get_solarposition(times, lat, lon)
    # # remove nighttime
    # solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    # fig, ax = plt.subplots(figsize=(12, 4))
    #
    # points = ax.scatter(solpos.azimuth, solpos.apparent_elevation, s=2,
    #                     c=solpos.index.dayofyear, label=None)
    # # fig.colorbar(points)
    #
    # for hour in np.unique(solpos.index.hour):
    #     # choose label position by the largest elevation for each hour
    #     subset = solpos.loc[solpos.index.hour == hour, :]
    #     height = subset.apparent_elevation
    #     pos = solpos.loc[height.idxmax(), :]
    #     ax.text(pos['azimuth'], pos['apparent_elevation'], str(hour))
    #
    # for date in pd.to_datetime(['2019-03-21', '2019-06-21', '2019-12-21', datetime.now().strftime("%d-%m-%y")]):
    #     times = pd.date_range(date, date + pd.Timedelta('24h'), freq='5min', tz=tz)
    #     solpos = solarposition.get_solarposition(times, lat, lon)
    #     solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
    #     label = date.strftime('%Y-%m-%d')
    #     ax.plot(solpos.azimuth, solpos.apparent_elevation, label=label)
    #
    # ax.scatter(solarposition.get_solarposition(datetime.utcnow(), lat, lon).azimuth,
    #            solarposition.get_solarposition(datetime.utcnow(), lat, lon).apparent_elevation, s=1200, label='now')
    # t = 0
    # if t == 0:
    #     im = plt.imread("mpei_pano2_cmpr.jpg")
    #
    # implot = plt.imshow(im, origin='upper', extent=[0, 360, 0, 21])
    #
    # # plt.scatter([100,8000], [0, 1500])
    #
    # # ax.plot(x, x, '--', linewidth=5, color='firebrick')
    # # ax.figure.legend(loc='upper left')
    # ax.set_xlabel('Solar Azimuth (degrees)')
    # ax.set_ylabel('Solar Elevation (degrees)')
    # #ax.set_aspect('auto')
    # # Convert plot to PNG image
    # pngImage = io.BytesIO()
    # plt.savefig('mpei_pan.png')
    # FigureCanvas(fig).print_png(pngImage)
    #
    # # Encode PNG image to base64 string
    # pngImageB64String = "data:image/png;base64,"
    # pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    #<!><center><a href="https://ibb.co/BBVk3vx"><img src="https://i.ibb.co/rHG90qS/PANO-20211008-124808-2.jpg" alt="PANO-20211008-124808-2" border="1"></a></center><-->


    try:
        mpei_pv_24 = pd.read_sql_query("SELECT * FROM (SELECT * FROM mpei_ses.invertor_dump WHERE DATE_FORMAT(mpei_ses.invertor_dump.DATE, '%%Y-%%m-%%d') >= CURDATE()) sub ORDER BY 'INDEX' ASC;", mydb) #("SELECT * FROM (SELECT * FROM mpei_meteo.meteo_dump_solar_temp_press ORDER BY 'INDEX' DESC LIMIT 86400) sub ORDER BY 'INDEX' ASC;", mydb)
        mpei_pv_24 = mpei_pv_24.set_index(pd.DatetimeIndex(mpei_pv_24['DATE']))
        #meteo24 = meteo24.resample('1Min').mean()



        # Generate plot

        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])

        pio.templates["plotly_dark_custom"] = pio.templates["plotly_dark"]

        pio.templates["plotly_dark_custom"].update({

            'layout_paper_bgcolor': 'rgba(0,0,0,0)',
            'layout_plot_bgcolor': 'rgba(0,0,0,0)'
        })



        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['output_power(Вт)']/1000,
                       name="Выходная мощность"),
            secondary_y=False, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv1_input_power(Вт)']/1000,
                       name="Входная мощность 1"),
            secondary_y=False, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv1_input_power(Вт)']/1000,
                       name="Входная мощность 2"),
            secondary_y=False, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['grid_voltage(В)'],
                       name="Напряжение сети"),
            secondary_y=False, row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['grid_frequency(Гц)'],
                       name="Частота"),
            secondary_y=True, row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv1_voltage(В)'],
                       name="Входное напряжение 1"),
            secondary_y=False, row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv2_voltage(В)'],
                       name="Входное напряжение 2"),
            secondary_y=False, row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv1_current(А)'],
                       name="Входной ток 1"),
            secondary_y=True, row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=mpei_pv_24.index, y=mpei_pv_24['pv2_current(А)'],
                       name="Входной ток 2"),
            secondary_y=True, row=3, col=1
        )



        fig.update_layout(margin=dict(l=20, r=20), height=1200, template = 'plotly_dark_custom',
                          title_text="Запрос данных мониторинга сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"), title_x=0.5,
                          legend=dict(
                              orientation="h", xanchor="center", x=0.5, y=1.05)
                          )



        fig.update_xaxes(title_text="Дата", row=1, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=1)
        fig.update_xaxes(title_text="Дата", row=3, col=1)
        fig.update_yaxes(title_text="<b>Мощность</b>, кВт", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="<b>Напряжение</b>, В", secondary_y=False, row=2, col=1)
        fig.update_yaxes(title_text="<b>Частота</b>, Гц", secondary_y=True, row=2, col=1)
        fig.update_yaxes(title_text="<b>Напряжение</b>, В", secondary_y=False, row=3, col=1)
        fig.update_yaxes(title_text="<b>Ток</b>, А", secondary_y=True, row=3, col=1)



        plt.style.use('seaborn-white')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('mpei_pv.html', graphJSON=graphJSON) #image=pngImageB64String


    except Exception as e:
        print(e)
        return render_template("mpei_error.html")


@app.route('/mpei_meteo')
def mpei():
    tz = 'Europe/Moscow'
    lat, lon = 55.75497, 37.709
    times = pd.date_range('2020-01-01 00:00:00', '2021-01-01', closed='left',
                          freq='H', tz=tz)

    try:
        meteo24 = pd.read_sql_query("SELECT * FROM (SELECT * FROM mpei_meteo.meteo_dump_solar_temp_press WHERE DATE_FORMAT(mpei_meteo.meteo_dump_solar_temp_press.DATE, '%%Y-%%m-%%d') >= CURDATE()) sub ORDER BY 'INDEX' ASC;", mydb) #("SELECT * FROM (SELECT * FROM mpei_meteo.meteo_dump_solar_temp_press ORDER BY 'INDEX' DESC LIMIT 86400) sub ORDER BY 'INDEX' ASC;", mydb)
        meteo24 = meteo24.set_index(pd.DatetimeIndex(meteo24['DATE']))
        meteo24 = meteo24.resample('1Min').mean()



        # Generate plot

        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])

        pio.templates["plotly_dark_custom"] = pio.templates["plotly_dark"]

        pio.templates["plotly_dark_custom"].update({

            'layout_paper_bgcolor': 'rgba(0,0,0,0)',
            'layout_plot_bgcolor': 'rgba(0,0,0,0)'
        })



        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['TA'],
                       name="Температура воздуха"),
            secondary_y=True, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['SR_1M'],
                       name="Суммарная солнечная радиация,<br>на горизонтальной поверхности"),
            secondary_y=False, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['SR_45_1M'],
                       name="Суммарная солнечная радиация,<br>на наклонной поверхности"),
            secondary_y=False, row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['RH'],
                       name="Относительная влажность"),
            secondary_y=False, row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['PA'],
                       name="Атмосферное давление"),
            secondary_y=True, row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=meteo24.index, y=meteo24['PR'],
                       name="Количество осадков"),
            secondary_y=False, row=3, col=1
        )

        # fig.add_trace(
        #    go.Scatter(x=tmy.index.dayofyear, y=tmy['dni'], name="Прямая солнечная радиация<br>на нормальной поверхности"),
        #    secondary_y=True,
        # )

        fig.update_layout(margin=dict(l=20, r=20), height=1200, template = 'plotly_dark_custom',
                          title_text="Запрос данных мониторинга сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"), title_x=0.5,
                          legend=dict(
                              orientation="h", xanchor="center", x=0.5, y=1.05)
                          )



        fig.update_xaxes(title_text="Дата", row=1, col=1)
        fig.update_xaxes(title_text="Дата", row=2, col=1)
        fig.update_xaxes(title_text="Дата", row=3, col=1)
        fig.update_yaxes(title_text="<b>Температура воздуха</b>, °С", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="<b>Солнечная радиация</b>, Вт/кв.м", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="<b>Относительная влажность</b>, %", secondary_y=False, row=2, col=1)
        fig.update_yaxes(title_text="<b>Атмосферное давление</b>, мм.рт.ст", secondary_y=True, row=2, col=1)
        fig.update_yaxes(title_text="<b>Количество осадков</b>, мм", secondary_y=False, row=3, col=1)
        fig.update_xaxes(mirror=True)
        fig.update_xaxes(gridcolor= '#81868f')



        plt.style.use('seaborn-white')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('mpei_pv.html', graphJSON=graphJSON, temp=meteo24['TA'], ghi=meteo24['SR_1M'],
                               gti=meteo24['SR_45_1M'],
                               labels = meteo24.index, legend = 'temperature') #image=pngImageB64String


    except Exception as e:
        print(e)
        return render_template("mpei_error.html")






@app.route('/user/<string:name>/<int:id>')
def user (name, id):
    return "User page: " + name + " - " + str(id)


@app.route("/nvchb", methods=["GET"])
def plotView():

    #print(pd.read_sql_query('SELECT * FROM ' + 'weather' + '.' + 'yrno_nvchb1' + ';', mydb))
    try:
        freshest_from_sql = pd.read_sql_query('SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;', mydb)
        freshest_from_sql = freshest_from_sql.set_index(pd.DatetimeIndex(freshest_from_sql['DateTime']))


        from_sql_yrno01utc = pd.read_sql_query('SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 01 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;', mydb)
        from_sql_yrno01utc = from_sql_yrno01utc.set_index(pd.DatetimeIndex(from_sql_yrno01utc['DateTime']))
        yrno01utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno01utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno01utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno01utc,
                                           56.11, 47.48, 'Europe/Moscow', 40, 180)

        from_sql_yrno09utc = pd.read_sql_query(
            'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 09 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
            mydb)
        from_sql_yrno09utc = from_sql_yrno09utc.set_index(pd.DatetimeIndex(from_sql_yrno09utc['DateTime']))
        yrno09utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno09utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno09utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno09utc,
                                                            56.11, 47.48, 'Europe/Moscow', 40, 180)
        #
        # #подумай о 48-ми часовом прогнозе (рынок на сутки вперед, заявки подаются в 13:30 на след.день)
        from_sql_yrno13utc = pd.read_sql_query(
            'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 13 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
            mydb)
        from_sql_yrno13utc = from_sql_yrno13utc.set_index(pd.DatetimeIndex(from_sql_yrno13utc['DateTime']))
        yrno13utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno13utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno13utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno13utc,
                                                            56.11, 47.48, 'Europe/Moscow', 40, 180)
        #
        # from_sql_yrno20utc = pd.read_sql_query(
        #     'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 20 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
        #     mydb)
        # from_sql_yrno20utc = from_sql_yrno09utc.set_index(pd.DatetimeIndex(from_sql_yrno20utc['DateTime']))
        # yrno20utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno20utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        # yrno20utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno20utc,
        #                                                     56.11, 47.48, 'Europe/Moscow', 40, 180)

        freshest_fcst = sun_forecast.get_forecast_from_sql(freshest_from_sql,
                                           56.11, 47.48, 'Europe/Moscow', 40, 180)
        freshest_fcst_dt = pd.DatetimeIndex(freshest_from_sql['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]


        # Generate plot
        plt.style.use('seaborn-white')
        fig = Figure(figsize=(8, 5))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("Прогноз сделан в " + freshest_fcst_dt)
        axis.set_xlabel("Дата и время")
        axis.set_ylabel("Среднечасовая интенсивность солнечной \nрадиации, Вт/кв.м")

        fig2 = Figure(figsize=(8, 5))
        axis2 = fig2.add_subplot(1, 1, 1)
        axis2.set_title("Запрос данных мониторинга сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"))
        axis2.set_xlabel("Дата и время")
        axis2.set_ylabel("Интенсивность солнечной \nрадиации, Вт/кв.м")

        #forecast_itself = sun_forecast.get_forecast()[0]
        #sun_forecast.get_forecast()[1].to_csv('raw.csv')

        #ntc_data = ntc_data_loader()
        #print(ntc_data)
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

        axis.plot(freshest_fcst[0], marker='o')
        axis.xaxis_date()
        axis.grid()
        #axis.plot(range(5), range(5), "ro-")

        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)

        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

        axis2.plot(freshest_fcst[0], label = 'Прогноз от ' + freshest_fcst_dt, marker='o')
        axis2.plot(yrno01utc_fcst[0], label='Прогноз от ' + yrno01utc_fcst_dt, marker='x')
        axis2.plot(yrno09utc_fcst[0], label='Прогноз от ' + yrno09utc_fcst_dt, marker='v')
        axis2.plot(yrno13utc_fcst[0], label='Прогноз от ' + yrno13utc_fcst_dt, marker='^')
        axis2.plot(ntc_data[0], label = "Факт. данные \n в исходном разрешении")
        axis2.plot(ntc_data[1], label="Факт. данные \n с часовым усреднением", marker='s')
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

        print(freshest_fcst[0])

        return render_template("nvchb.html", image=pngImageB64String, image2=pngImageB64String2,
                               tables=[pd.DataFrame(freshest_fcst[0]).to_html()],
                               titles=pd.DataFrame(freshest_fcst[0]).columns.values)

    except:
        abort(404)





@app.route("/nvchb2", methods=['POST', 'GET'])
def plotViewo():

    try:
        #print(pd.read_sql_query('SELECT * FROM ' + 'weather' + '.' + 'yrno_nvchb1' + ';', mydb))
        freshest_from_sql = pd.read_sql_query('SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;', mydb)
        freshest_from_sql = freshest_from_sql.set_index(pd.DatetimeIndex(freshest_from_sql['DateTime']))


        from_sql_yrno01utc = pd.read_sql_query('SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 01 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;', mydb)
        from_sql_yrno01utc = from_sql_yrno01utc.set_index(pd.DatetimeIndex(from_sql_yrno01utc['DateTime']))
        yrno01utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno01utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno01utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno01utc,
                                           56.11, 47.48, 'Europe/Moscow', 40, 180)

        from_sql_yrno09utc = pd.read_sql_query(
            'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 09 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
            mydb)
        from_sql_yrno09utc = from_sql_yrno09utc.set_index(pd.DatetimeIndex(from_sql_yrno09utc['DateTime']))
        yrno09utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno09utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno09utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno09utc,
                                                            56.11, 47.48, 'Europe/Moscow', 40, 180)
        #
        # #подумай о 48-ми часовом прогнозе (рынок на сутки вперед, заявки подаются в 13:30 на след.день)
        from_sql_yrno13utc = pd.read_sql_query(
            'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 13 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
            mydb)
        from_sql_yrno13utc = from_sql_yrno13utc.set_index(pd.DatetimeIndex(from_sql_yrno13utc['DateTime']))
        yrno13utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno13utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        yrno13utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno13utc,
                                                            56.11, 47.48, 'Europe/Moscow', 40, 180)

        # from_sql_yrno20utc = pd.read_sql_query(
        #     'SELECT * FROM (SELECT * FROM weather.yrno_nvchb1 WHERE HOUR(fcst_dt_utc) = 20 ORDER BY id DESC LIMIT 24) sub ORDER BY id ASC;',
        #     mydb)
        # from_sql_yrno20utc = from_sql_yrno09utc.set_index(pd.DatetimeIndex(from_sql_yrno20utc['DateTime']))
        # yrno20utc_fcst_dt = pd.DatetimeIndex(from_sql_yrno20utc['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]
        # yrno20utc_fcst = sun_forecast.get_forecast_from_sql(from_sql_yrno20utc,
        #                                                     56.11, 47.48, 'Europe/Moscow', 40, 180)

        freshest_fcst = sun_forecast.get_forecast_from_sql(freshest_from_sql,
                                           56.11, 47.48, 'Europe/Moscow', 40, 180)
        freshest_fcst_dt = pd.DatetimeIndex(freshest_from_sql['fcst_dt_utc']).strftime("%H:%M UTC %d-%m")[0]

        #ntc_data = ntc_data_loader()

        # Generate plot
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        #fig.add_trace(
           #go.Scatter(x=ntc_data[0].index,y=ntc_data[0][1], name="Факт. данные"),
           #secondary_y=False,
        #)

        #fig.add_trace(
           #go.Bar(x=ntc_data[1].index,y=ntc_data[1][1], name="Факт. среднечасовые данные"),
           #secondary_y=False,
        #)



        fig.add_trace(
            go.Scatter(x=freshest_fcst[0].index, y=freshest_fcst[0], name='Прогноз от ' + freshest_fcst_dt),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=yrno01utc_fcst[0].index, y=yrno01utc_fcst[0], name='Прогноз от ' + yrno01utc_fcst_dt),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=yrno09utc_fcst[0].index, y=yrno09utc_fcst[0], name='Прогноз от ' + yrno09utc_fcst_dt),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=yrno13utc_fcst[0].index, y=yrno13utc_fcst[0], name='Прогноз от ' + yrno13utc_fcst_dt),
            secondary_y=False,
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='White')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='White')

        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='White')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='White')

        fig.update_layout(margin=dict(l=20, r=20),
                          title_text="Запрос данных мониторинга сделан в " + datetime.now().strftime("%H:%M %d-%m-%Y"), title_x=0.5, legend=dict(
                orientation="h", xanchor="center", x=0.5, y=1.175))

        fig.update_xaxes(title_text="Дата и время")
        fig.update_yaxes(title_text="Среднечасовая интенсивность солнечной<br>радиации, Вт/кв.м", secondary_y=False)

        # fig = px.line(tmy['temp_air'], x=tmy.index.dayofyear, y='temp_air')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


        return render_template("nvchb.html", graphJSON=graphJSON)

    except:
        return render_template("nvchb_error.html")









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