import os
import mysql.connector
import pandas as pd
from forecasting import SunForecast
from datetime import datetime

db_username = os.environ.get('DB_USER')
db_password = os.environ.get('DB_PASS')
db_host = os.environ.get('DB_HOST')
db_schema = os.environ.get('DB_SCHEMA')
table_name = 'yrno_nvchb1'

print(db_host)

connection = mysql.connector.connect(host = db_host, user = 'b4166a59d44cf9',
                                     passwd = db_password, db = db_schema)

sun_forecast = SunForecast()



df = sun_forecast.get_forecast()[1].reset_index()
df['fcst_dt'] = datetime.utcnow().strftime('%y-%m-%d %H:%M:%S')

cols = ['fcst_dt', 'time', 'air_pressure_at_sea_level', 'air_temperature', 'cloud_area_fraction',
       'cloud_area_fraction_high',
       'cloud_area_fraction_medium', 'cloud_area_fraction_low', 'dew_point_temperature',
       'fog_area_fraction', 'relative_humidity', 'ultraviolet_index_clear_sky',
        'wind_speed', 'wind_from_direction']

#print(df[cols])

df[cols].to_sql(table_name, con = connection)

print(pd.read_sql_query('SELECT * FROM ' + db_schema +'.' + table_name+ ';', connection))






#print(sun_forecast.get_forecast()[1][cols])
#print(sun_forecast.get_forecast()[1])