import os
import pymysql
import pandas as pd
from forecasting import SunForecast
from datetime import datetime
from sqlalchemy import create_engine


db_username = os.environ.get('DB_USER')
db_password = os.environ.get('DB_PASS')
db_host = os.environ.get('DB_HOST')
db_schema = os.environ.get('DB_SCHEMA')
table_name1 = 'yrno_nvchb1'
table_name2 = 'yrno_mpei1'


# connection = pymysql.connector.connect(host = db_host, user = 'b4166a59d44cf9',
#                                     password = db_password, db = db_schema)

sun_forecast = SunForecast()



df = sun_forecast.get_forecast()[1].reset_index()
df['fcst_dt_utc'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

cols = ['fcst_dt_utc', 'time', 'air_pressure_at_sea_level', 'air_temperature', 'cloud_area_fraction',
       'cloud_area_fraction_high',
       'cloud_area_fraction_medium', 'cloud_area_fraction_low', 'dew_point_temperature',
       'fog_area_fraction', 'relative_humidity', 'ultraviolet_index_clear_sky',
        'wind_speed', 'wind_from_direction']

#print(df[cols])

mydb = create_engine('mysql+pymysql://' + db_username + ':' + db_password + '@' + db_host + ':' + str(3306) + '/' + db_schema , echo=False)

print(df[cols])

df = df[cols].rename(columns={'time':'DateTime', 'air_pressure_at_sea_level':'air_pressure',
                              'cloud_area_fraction':'cloud_area_frac', 'cloud_area_fraction_high':'cloud_area_frac_high',
                              'cloud_area_fraction_medium':'cloud_area_frac_medium',
                              'cloud_area_fraction_low':'cloud_area_frac_low', 'dew_point_temperature':'dew_point_temp',
                              'fog_area_fraction':'fog_area_frac', 'relative_humidity':'rel_humidity',
                              'ultraviolet_index_clear_sky':'uv_idx_clear_sky', 'wind_from_direction':'wind_dir'})

df.to_sql(table_name1, con = mydb, if_exists='append', index=False)

print(pd.read_sql_query('SELECT * FROM ' + db_schema +'.' + table_name1+ ';', mydb))

sun_forecast.location_lat = 55.755
sun_forecast.location_lon = 37.709

df = sun_forecast.get_forecast()[1].reset_index()
df['fcst_dt_utc'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

cols = ['fcst_dt_utc', 'time', 'air_pressure_at_sea_level', 'air_temperature', 'cloud_area_fraction',
       'cloud_area_fraction_high',
       'cloud_area_fraction_medium', 'cloud_area_fraction_low', 'dew_point_temperature',
       'fog_area_fraction', 'relative_humidity', 'ultraviolet_index_clear_sky',
        'wind_speed', 'wind_from_direction']

df = df[cols].rename(columns={'time':'DateTime', 'air_pressure_at_sea_level':'air_pressure',
                              'cloud_area_fraction':'cloud_area_frac', 'cloud_area_fraction_high':'cloud_area_frac_high',
                              'cloud_area_fraction_medium':'cloud_area_frac_medium',
                              'cloud_area_fraction_low':'cloud_area_frac_low', 'dew_point_temperature':'dew_point_temp',
                              'fog_area_fraction':'fog_area_frac', 'relative_humidity':'rel_humidity',
                              'ultraviolet_index_clear_sky':'uv_idx_clear_sky', 'wind_from_direction':'wind_dir'})

df.to_sql(table_name2, con = mydb, if_exists='append', index=False)


#dff = pd.read_sql_query('SELECT * FROM ' + db_schema +'.' + table_name2+ ';', mydb)
#dff = dff.set_index('DateTime')
print(df)


#print(sun_forecast.get_forecast()[1][cols])
#print(sun_forecast.get_forecast()[1])