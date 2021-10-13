from flask import Flask, render_template, url_for, Response
import base64
from datetime import datetime
import pandas as pd
import io
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from forecasting import SunForecast

app = Flask(__name__)

sun_forecast = SunForecast()


@app.route('/')
@app.route('/home')
def index():
    return render_template('index.html')


@app.route('/mpei')
def info():
    return render_template('mpei.html')


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
    app.run(debug=True)