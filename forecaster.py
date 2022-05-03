import pandas as pd
import pickle
import csv
from requests import get
from datetime import datetime
from simplejson import loads
from pvlib.irradiance import aoi
from pvlib.solarposition import spa_python
from pvlib.pvsystem import retrieve_sam
from sklearn.preprocessing import MinMaxScaler
from time import localtime, strftime
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from bs4 import BeautifulSoup
import numpy as np
from time import localtime, strftime
from datetime import datetime, timedelta, timezone
import re
import requests

import sun_geo
sun=sun_geo.sunny()

class Forecaster():


    def yrnoparser(latitude, longitude, set_timezone, horizon=48, 
                useragent='mpei.ru, NarynbayevA@mpei.ru, student project'):
        
        """
        useragent (str) - UserAgent для авторизации в системе yr.no 
        latitude (float) - широта местности, для которой запрашивается прогноз
        longitude (float) - долгота местности, для которой запрашивает прогноз
        set_timezone (str) - часовой пояс в формате pytz
        horizon (float) - (min=2, max=91, default=24) - горизонт упреждения прогноза в часах
        """
        
        headers={'User-Agent':useragent}
        url = 'https://api.met.no/weatherapi/locationforecast/2.0/complete?lat='+ \
        str(latitude)+'&lon='+str(longitude)
        
        #запрос хэдеров со статистикой последних обновлений прогнозов
        try:
            if get('https://api.met.no/weatherapi/locationforecast/2.0/status',
                        headers=headers).status_code == 200:
                #сервер работает и UserAgent не заблокирован
                #local_time_now=datetime.utcnow().replace(tzinfo=UTC).astimezone(
                    #timezone(str(get_localzone()))) #присвоение часового пояса по времени на компьютере

                #подгрузка json-ов
                raw_forecast=loads(get(url, headers=headers).text)
                raw_DataFrame = pd.DataFrame.from_dict(raw_forecast.get('properties', 
                                                                        {}).get('timeseries'))

                #генерация pandas-датафрейма с прогнозом погоды
                listed_forecast = []
                for i in range(len(raw_forecast.get('properties', 
                                                    {}).get('timeseries',{'data'}))):
                    listed_forecast.append(
                        dict(raw_forecast['properties']['timeseries'][i])['data']['instant']['details'])

                forecast = pd.DataFrame(listed_forecast, raw_DataFrame['time'])
                forecast.index = pd.to_datetime(forecast.index)
                forecast = forecast.tz_localize(None)
                forecast = forecast.tz_localize('UTC').tz_convert(str(set_timezone))
                forecast[['zenith','solar_azimuth']] = spa_python(forecast.index,
                                                                latitude,
                                                                longitude)[['apparent_zenith',
                                                                            'azimuth']]
            else:
                return('Ошибка! Сервер не отвечает, либо доступ к API запрещен')
        except Exception as e:
            return ('Неизвестная ошибка', e)
        return forecast.iloc[1:horizon+1] #возвращает датайфрейм с прогнозом


    def rp5_parser(self, url, lat, lon):

        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Cache-Control': 'max-age=0',
            'Connection': 'keep-alive', 'Host': 'rp5.ru', 'Sec-Fetch-Dest': 'document', 'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1', 'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
            'cookie': 'PHPSESSID=b52677c7a0c53e6bcac0cbccda01a6f2; __utmc=66441069; __utmz=66441069.1586692831.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); located=1; extreme_open=false; __gads=ID=875638f002ac7f69:T=1586693599:S=ALNI_MbIIJzruHefdR7SWM-U-iaFDg6hfw; ftab=3; full_table=1; __utma=66441069.536792610.1586692831.1586792796.1586981798.6; i=6058%7C6058%7C6058%7C6058%7C6058; iru=6058%7C6058%7C6058%7C6058%7C6058; ru=%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA; last_visited_page=http%3A%2F%2Frp5.ru%2F%D0%9F%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0_%D0%B2_%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%D0%B5; lang=ru; __utmb=66441069.2.10.1586981798; is_adblock=0'
        }

        try:
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'lxml')
            table = soup.find_all('table')[10]  # таблица с прогнозом на сутки с rp5
            table1 = soup.find('table', id='forecastTable_1_3')
            # парсинг HTML таблицы с rp5
            lst = np.zeros((14, 25)).tolist()
            listo = []
            for i in range(0, 40):  # 26
                listo.append([])
            lst_counter = 0
            for row in table1.find_all('tr'):

                for cell in row.find_all('td'):
                    listo[lst_counter].append(cell.text)
                lst_counter += 1
            day = []
            date = []
            timeh = pd.Series(listo[1][1:40]).astype(int)  # 25
            today = datetime.now().strftime('%Y-%m-%d')
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            for hour in timeh:
                if hour > localtime().tm_hour and hour > 0:
                    date.append(
                        today + ' ' + str(hour) + ':00')  # собираем человеческий формат даты вместо того, что на RP5
                    day.append(localtime().tm_yday)
                elif hour <= localtime().tm_hour and hour >= 0:
                    date.append(tomorrow + ' ' + str(hour) + ':00')
                    day.append(localtime().tm_yday + 1)

            cld = []
            cld_raw = []

            for elem in table1.findAll("div", {"class": "cc_0"}):
                for tag in elem.find_all(onmouseover=True):
                    #print(tag)
                    jasno = re.findall(r'\b[Я]\w+', tag['onmouseover'])
                    clouds_info_raw_text = re.findall(r'\b\w+',
                                                      tag[
                                                          'onmouseover'])  # сырая информация с детализацией по облачности
                    cld_raw.append(clouds_info_raw_text)
                    if jasno == ['Ясно'] and re.findall(r'\d{1,4}', tag['onmouseover']) == []:
                        cld.append([0])
                        # print(re.findall(r'\b[Я]\w\w\w',tag['onmouseover']))
                    #         print(i)
                    if re.findall(r'\d{1,4}%', tag['onmouseover']) != []:
                        cld.append([int(s) for s in (
                            re.findall(r'\d{1,4}',
                                       tag['onmouseover']))])  # добавление значений облачности в процентах в список
                        # print([int(s) for s in (re.findall(r'\d{1,4}',tag['onmouseover']))])

            clouds = []

            for el in cld:
                # clouds.append(max(el))  # выбор максимального значения процента облачности
                if sum(el) >= 100:
                    clouds.append(100)
                else:
                    clouds.append(sum(el))

            # print(listo)

            for i in range(0, len(listo)):
                if 'Температура' in str(listo[i]):
                    temp_idx = i
                if 'Давление' in str(listo[i]):
                    pressure_idx = i
                if 'Ветер' in str(listo[i]):
                    wind_idx = i
                if 'Влажность' in str(listo[i]):
                    hum_idx = i

            temp = []
            for i in listo[
                temp_idx]:  # индекс listo - номер строки из таблицы прогноза с rp5 с соответвующим ему параметром
                temp.append(re.split(r'\s', i)[1])
            pressure = []
            # print(listo[pressure_idx])
            for i in listo[pressure_idx]:
                pressure.append(re.split(r'\s', i)[1])
            wind = []
            for i in listo[wind_idx]:
                if len(re.split(r'\s', i)) == 1:
                    j = 0
                else:
                    j = 1
                wind.append(re.split(r'\s', i)[j])

            temperature = pd.Series(temp[1:37])  # 25
            pressureHPA = pd.Series(pressure[1:37]).astype(float) * 1.33322  # 25
            pressure = pd.Series(pressure[1:37])  # 25
            wind = pd.Series(wind[1:37])  # 25
            clouds = pd.Series(clouds)
            humidity = pd.Series(listo[hum_idx][1:37])  # 25
            date = pd.Series(date)
            day = pd.Series(day)
            #lat = 56.0827338
            inc = sun.inc_a(lat, day, timeh, 40, 1)[1]
            zen = sun.zen(lat, day, timeh)[0]
            forecastRP5 = pd.concat([date, inc, temperature, clouds, pressureHPA, humidity, wind, zen, pressure],
                                    axis=1).dropna()
            forecastRP5.rename(
                columns={forecastRP5.columns[0]: 'Date', forecastRP5.columns[1]: 'inc', forecastRP5.columns[2]: 'T',
                         forecastRP5.columns[3]: 'Cl', forecastRP5.columns[4]: 'P0', forecastRP5.columns[5]: 'U',
                         forecastRP5.columns[6]: 'Ff', forecastRP5.columns[7]: 'zen',
                         forecastRP5.columns[8]: 'P0mm'}, inplace=True)
            # forecastRP5['Date'] = pd.to_datetime(forecastRP5['Date'])

            artificial_datetime = pd.date_range(datetime.now() + timedelta(hours=1),
                                                datetime.now() + timedelta(hours=36),
                                                freq='h').strftime("%y-%m-%d %H")
            artificial_datetime1 = pd.to_datetime(artificial_datetime).strftime("%y-%m-%d %H:%M:%S")

            forecastRP5['Date'] = artificial_datetime1

            forecastRP5['Clouds_raw_info'] = cld_raw[0:36]

            forecastRP5['Clouds_raw_info'] = forecastRP5['Clouds_raw_info'].apply(', '.join)

            forecastRP5 = forecastRP5.set_index('Date')

            forecastRP5[['zenith', 'solar_azimuth']] = spa_python(forecastRP5.index,
                                                               lat,
                                                               lon)[['apparent_zenith',
                                                                           'azimuth']]

            return forecastRP5
        except Exception as e:
            return ('Ошибка парсинга rp5', e)





    def mkforecast(filename, weather_fcst, columns, X_scaler_param_file,
                Y_scaler_param_file, pv_tilt, pv_azimuth, test_mode, savefile = False):
        """
        filename (str) - путь к pickle-файлу модели прогноза
        weather_fcst (pandas Dataframe) - датафрейм с данными метеопрогноза
        columns (list of str) - список наименований столбцов из датайфрейма с предикторами
        X_scaler_param_file (scaler file) - путь к файлу с параметрами масштабирования предикторов
        Y_scaler_param_file (scaler file) - путь к файлу с параметрами масштабирования предиктантов  
        pv_tilt (int) - угол наклона фотоэлектрических модулей к горизонту (град.)
        pv_azimuth (int) - азимутальный угол фотоэлектрических модулей (град.)
        savefile (bool) - флажок сохранения файла прогноза
        """
        model=pickle.load(open(filename, 'rb')) #загрузка модели из pickle-файла
 
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        
        X_scaler.min_= pickle.load(open(X_scaler_param_file, 'rb'))[0]

        X_scaler.scale_= pickle.load(open(X_scaler_param_file, 'rb'))[1]
        
        Y_scaler.min_= pickle.load(open(Y_scaler_param_file, 'rb'))[0]
        
        Y_scaler.scale_= pickle.load(open(Y_scaler_param_file, 'rb'))[1]
       
        if test_mode == False:
            weather_fcst['aoi'] = aoi(pv_tilt, pv_azimuth,
                                weather_fcst['zenith'], weather_fcst['solar_azimuth'])
            X = X_scaler.transform(weather_fcst[columns])
            #неотмасштабированный ряд предсказаний солнечной радиации
            raw_I_fcst=model.predict(X).reshape(len(model.predict(X)), -1)  
            I_fcst=Y_scaler.inverse_transform(raw_I_fcst) #обратное масштабирование ряда предсказаний 
            I_fcst[I_fcst < 0] = 0
            weather_fcst['irradiance'] = I_fcst
            if savefile==True:
                weather_fcst['irradiance'].to_csv('Прогноз, сделанный в '+
                    strftime("%H-%M %d-%m-%Y", localtime())+'.csv')
            return weather_fcst[['irradiance']] #'air_temperature',
        else:
            X = X_scaler.transform(weather_fcst[columns])
            raw_I_fcst=model.predict(X).reshape(len(model.predict(X)), -1)  
            I_fcst=Y_scaler.inverse_transform(raw_I_fcst)  
            I_fcst[I_fcst < 0] = 0
            weather_fcst['irradiance'] = I_fcst
            return weather_fcst[['T', 'irradiance']]     

    #функция расчета нормализованной среднеквадратичной ошибки
    def nrmse(y_real, y_pred):
        #нормализованная среднеквадратичная ошибка в процентах
        return 100*mean_squared_error(y_real,y_pred,squared=False) / (y_real.mean())

    def split_data(X, y, test_size, shuffle=False):
        
        """
        Деление выборки на обучающую и тестовую
        X (pandas DataFrame or array-like) - массив предикторов
        y (pandas Series or array-like) - массив или датафрейм-вектор предиктанта
        test_size (float) - доля тестовой выборки (рекомендуется в диапазоне от 0.2 до 0.3)
        shuffle (bool) - флажок перемешивания данных в массиве 
        """
        
        X_train0, X_test0, y_train0, y_test0 = train_test_split(X, y, 
                                                                test_size=0.25, 
                                                                random_state=1, 
                                                                shuffle=False) #доля тестовой выборки - 25%
        #преобразование датафреймов в массивы типа numpy
        X_train=X_train0.to_numpy()
        y_train=y_train0.to_numpy().reshape(len(y_train0), -1)
        X_test=X_test0.to_numpy()
        y_test=y_test0.to_numpy().reshape(len(y_test0), -1)
        
        return X_train, y_train, X_test, y_test
        
    def scale_data(data, filename='scaler_params', save_to_file = True):
        """
        Функция машстабирования данных (актуально для метрических алгоритмов)
        data (array-like) - массив данных для масштабирования
        filename (str) - путь к экспортируемому файлу с параметрами масштабирования
        save_to_file (bool) - флажок сохранения файла с параметрами масштабирования
        """
        scaler= MinMaxScaler()
        #Масштабируем данные
        scaled = scaler.fit_transform(data)
        if save_to_file is True:
            pickle.dump([scaler.min_, scaler.scale_], open(str(filename), 'wb'))
        
        return scaled, scaler.min_, scaler.scale_


    def fit_mlp(X, y, hidden_layer_sizes, trained_model_filename, savemodel=True):
        """
        Функция обучения нейронной сети типа многослойный перцептрон.
        Далее перечислены параметры MLPRegressor, которые при необходимости можно добавить вручную (hidden_layer_sizes=100, activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        
        X (pandas DataFrame or array-like) - обучающий массив предикторов
        y (pandas Series or array-like) - обучающий массив или датафрейм-вектор предиктанта
        hidden_layer_sizes (int or tuple) - число скрытых слоев и нейронов в них. Например, два скрытых слоя с 8 и 64 нейронами - (8,64)
        savemodel (bool) - флажок сохранения файла модели
        """
        MLP = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,verbose=False, random_state=1) #остальные параметры настроены по умолчанию
        MLP.fit(X, y.ravel())
        
        if savemodel == True:
            pickle.dump(MLP, open(trained_model_filename, 'wb'))
        
        

    def fit_rf(X, y, trees=100, savemodel=True):
        """
        Функция обучения модели случайного леса.
        Далее перечислены параметры MLPRegressor, которые при необходимости можно добавить вручную (n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
        
        X (pandas DataFrame or array-like) - обучающий массив предикторов
        y (pandas Series or array-like) - обучающий массив или датафрейм-вектор предиктанта
        trees (int) - число решающих деревьев в лесу
        savemodel (bool) - флажок сохранения файла модели
        """
        
        RF = RandomForestRegressor(n_estimators = trees, max_depth=5, random_state=1) #max_depth - глубина развития деревьев решений
        RF.fit(X, y.ravel())
        
        if savemodel == True:
            pickle.dump(RF, open('rf_model.mdl', 'wb'))
            


    def training_summary(self,model,X_train,y_train,X_test,y_test,
                Xscaler_min, Xscaler_scale, Yscaler_min, Yscaler_scale): 
        
        """
        Функция для расчета точности обученной модели на тестовой подвыборке исходного датасета
        X_train (array) - обучающий массив предикторов
        X_test (array) - тестовый массив предикторов
        y_train (array) - обучающий массив предиктанта
        y_test (array) - тестовый массив предиктанта
        Xscaler_min (list or array-like) - параметр минимума масштабируемых предикторов
        Xscaler_scale (list or array-like) - параметр масштаба масштабируемых предикторов
        Yscaler_min (list or array-like) - параметр минимума масштабируемого предиктанта
        Yscaler_scale (list or array-like) - параметр масштаба масштабируемого предиктанта
        plott (bool) - флажок построения графиков
        metrics (bool) - флажок отображения метрик модели
        """  

        scalerX = MinMaxScaler()
        scalerY = MinMaxScaler()
        scalerX.min_ = Xscaler_min
        scalerX.scale_ = Xscaler_scale
        scalerY.min_ = Yscaler_min
        scalerY.scale_ = Yscaler_scale
        
        prediction=model.predict(X_test) #предсказание по тестовым признакам
        #prediction0=model.predict(X_train) #предсказание по обучающим признакам
        #y_pred0=prediction0.reshape(len(prediction0), -1)
        Ipred=prediction.reshape(len(prediction), -1)
        #на выходе модели выдается масштабированная величина I, которую для лучшей интерпретации результатов необходимо преобразовать обратно в именованную
        #ytrain=scalerY.inverse_transform(y_train) #обратное преобразование фактических величин I из обучающей выборки
        #y_pred_train=scalerY.inverse_transform(y_pred0) #обратное преобразование спрогнозированных величин I по обучающей выборке
        y_pred=scalerY.inverse_transform(Ipred) #обратное преобразование спрогнозированных величин I 
        y_real=scalerY.inverse_transform(y_test) #обратное преобразование фактических величин I
        
        return round(r2_score(y_real,y_pred),3), round(mean_absolute_error(y_real,y_pred),2), round(self.nrmse(y_real,y_pred),2)

    def PV_array_yield(pv_module_STC, pv_module_area, pv_kt, G_POA, Ta, n):
        """
        Функция расчета выходной мощности массива фотоэлектрических модулей

        G_POA (float, array-like, pandas DataFrame) - интенсивность солнечной радиации в плоскости модулей, Вт/кв.м
        Ta (float, array-like, pandas DataFrame) - температура воздуха, гр. Ц
        n (int) - количество модулей
        F (float) - площадь модуля
        eff_ref (float) - коэффициент полезного действия модуля при стандартных условиях (STC)
        KT (float) - температурный коэффициент мощности (%/гр. Ц) [необходимо уточнять в тех.паспорте модуля]
        """
        #pv_module = retrieve_sam(path=pv_modules_database).T.iloc[pv_type] #модель ФЭМ
        eff_ref = pv_module_STC / (1000 * pv_module_area) #КПД ФЭМ
        EffPV = ((Ta - 25) * pv_kt / 100 + 1) * eff_ref
        N_PV = G_POA * n * pv_module_area * EffPV
        return N_PV

    def train_new_model(self, train_dataset, predictors, predictand, X_scaler_output_filename, y_scaler_output_filename, new_trained_model_filename,
                        save_to_file=True):
        """В датафрейме содержатся следующие данные: интенсивность солнечной радиации (I, Вт/кв.м)
        на наклонной плоскости (пиранометр установлен в плоскости фотоэлектрических модулей (ФЭМ)),
        угол падения солнечной радиации на поверхность ФЭМ (aoi, град.), температура воздуха (T, гр.Ц),
        относительная влажность (U, %), общая облачность (Cl, %) для условий Новочебоксарска. Датафрейм был получен путем
        объединения (по индексу даты и времени) двух разных массивов данных, 
        состоящих из данных измерений солнечной радиации
        и архива метеоданных. 
        """
        #Настройка парсера даты, зависящего от формата дат в массиве исходных данных
        dateparse1 = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        #Считывание файла исходных данных
        data=pd.read_csv(train_dataset,parse_dates=[0],date_parser=dateparse1,dayfirst=True).set_index('Unnamed: 0')

        

        #Фильтрация ночных часов
        Xy = data.loc[data['aoi']<90] #угол падения солнечного излучения больше 

        #Выделяем предикторы (признаки, или аргументы аппроксимируемой функции) в отдельный датафрейм
        X = Xy[list(predictors)]

        #Выделяем целевую величину в отдельный датафрейм-вектор
        y = Xy[list(predictand)]

        
        #Деление выборки на обучающую и тестовую
        #X (pandas DataFrame or array-like) - массив предикторов
        #y (pandas Series or array-like) - массив или датафрейм-вектор предиктанта
        #test_size (float) - доля тестовой выборки (рекомендуется в диапазоне от 0.2 до 0.3)
        #shuffle (bool) - флажок перемешивания данных в массиве 

        #Разбиение массива данных на обучающую и тестовую выборки
        X_train, y_train, X_test, y_test = self.split_data(X, y, 0.25)

        #Масштабирование выборок (приведение к шкале от 0 до 1)
        X_train, X_train_scaler_min, X_train_scaler_scale = self.scale_data(X_train, X_scaler_output_filename,
                                                               save_to_file = True)
        y_train, y_train_scaler_min, y_train_scaler_scale = self.scale_data(y_train,
                                                              y_scaler_output_filename,
                                                               save_to_file = True)
        
        #Масштабируем тестовую выборку
        X_test = self.scale_data(X_test, save_to_file = False)[0]
        y_test = self.scale_data(y_test, save_to_file = False)[0]

        #Обучаем MLP-модель (многослойный перцептрон - по умолчанию - трехслойная нейронная сеть с 100 нейронами
        #в скрытом слое (остальные параметры модели (скорость обучения, алгоритм оптимизации и т.д.)
        #выбраны по умолчанию. На вход подаются отмасштабированные предикторы, на выходе - предсказанная целевая величина,
        #которую затем нужно масштабировать обратно в именованные единицы). Более тонкую настройку параметров функции
        #можно произвести путем изменения функции fit_mlp в forecaster.py. Также доступна модель случайного леса
        #(функция fir_rf)

        self.fit_mlp(X_train, y_train, 100, new_trained_model_filename, savemodel=True) #100 - число нейронов в одном скрытом слое. Для создания многослойного перцептрона вместо 100 нужно задать tuple, например (100, 200)


    def add_measurements_data(self, datetime, latitude, longitude, pv_tilt_angle, pv_azimuth_angle, data_to_add, dataset_for_adding):

        zenith = spa_python(datetime,latitude,longitude)['apparent_zenith']
        solar_azimuth = spa_python(datetime,latitude,longitude)['azimuth']

        angle_of_incidence = aoi(pv_tilt_angle, pv_azimuth_angle,
                                zenith, solar_azimuth)

        data_to_add.append(angle_of_incidence.values[0])

        with open(dataset_for_adding, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data_to_add)
                                                                


  

        