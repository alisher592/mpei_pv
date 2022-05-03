import time
start = time.process_time()
from bs4 import BeautifulSoup
from time import localtime, strftime
from datetime import datetime, timedelta, timezone
from tzlocal import get_localzone
import requests
#import yadisk
import pandas as pd
import numpy as np
import re
import pickle
import simplejson as json
import email.utils as eut
from sklearn import neural_network
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import pytz
import os
import sun_geo



sun=sun_geo.sunny()




url = r'https://rp5.ru/%D0%9F%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0_%D0%B2_%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%D0%B5'
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7', 'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive', 'Host': 'rp5.ru', 'Sec-Fetch-Dest': 'document', 'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1', 'Upgrade-Insecure-Requests': '1', 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
    'cookie': 'PHPSESSID=b52677c7a0c53e6bcac0cbccda01a6f2; __utmc=66441069; __utmz=66441069.1586692831.1.1.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); located=1; extreme_open=false; __gads=ID=875638f002ac7f69:T=1586693599:S=ALNI_MbIIJzruHefdR7SWM-U-iaFDg6hfw; ftab=3; full_table=1; __utma=66441069.536792610.1586692831.1586792796.1586981798.6; i=6058%7C6058%7C6058%7C6058%7C6058; iru=6058%7C6058%7C6058%7C6058%7C6058; ru=%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%7C%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA; last_visited_page=http%3A%2F%2Frp5.ru%2F%D0%9F%D0%BE%D0%B3%D0%BE%D0%B4%D0%B0_%D0%B2_%D0%9D%D0%BE%D0%B2%D0%BE%D1%87%D0%B5%D0%B1%D0%BE%D0%BA%D1%81%D0%B0%D1%80%D1%81%D0%BA%D0%B5; lang=ru; __utmb=66441069.2.10.1586981798; is_adblock=0'
}
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text,'lxml')
table = soup.find_all('table')[10] #таблица с прогнозом на сутки с rp5
table1=soup.find('table', id='forecastTable_1_3')
#парсинг HTML таблицы с rp5
lst=np.zeros((14,25)).tolist() 
listo=[]
for i in range(0,38):
    listo.append([])
lst_counter=0
for row in table1.find_all('tr'):
    
    for cell in row.find_all('td'):
        listo[lst_counter].append(cell.text)
    lst_counter+=1
day=[]
date=[]
timeh=pd.Series(listo[1][1:37]).astype(int)
today=datetime.now().strftime('%Y-%m-%d')

today_modified = (datetime.now()+timedelta(hours=1)).strftime('%Y-%m-%d %H')
print(today_modified)
print(pd.date_range(start=today_modified, periods=36, freq='H'))

tomorrow=(datetime.now()+ timedelta(days=1)).strftime('%Y-%m-%d')
for hour in timeh:
    if hour>localtime().tm_hour and hour>0:
        date.append(today + ' ' + str(hour) +':00') #собираем человеческий формат даты вместо того, что на RP5
        day.append(localtime().tm_yday)
    elif hour<=localtime().tm_hour and hour>=0:
        date.append(tomorrow + ' ' + str(hour) +':00')
        day.append(localtime().tm_yday+1)

cld=[]
for elem in table1.findAll("div", {"class": "cc_0"}):
    for tag in elem.find_all(onmouseover=True):
        #print(tag)
        jasno=re.findall(r'\b[Я]\w+',tag['onmouseover'])
        #print(jasno)
        if jasno==['Ясно'] and re.findall(r'\d{1,4}',tag['onmouseover'])==[]:
            cld.append([0])
            #print(re.findall(r'\b[Я]\w\w\w',tag['onmouseover']))
    #         print(i)
        if re.findall(r'\d{1,4}%',tag['onmouseover'])!=[]:
            cld.append([int(s) for s in (re.findall(r'\d{1,4}',tag['onmouseover']))]) #добавление значений облачности в процентах в список
            #print([int(s) for s in (re.findall(r'\d{1,4}',tag['onmouseover']))])


clouds=[]
for el in cld:
    clouds.append(max(el)) #выбор максимального значения процента облачности

#print(listo)

for i in range(0,len(listo)):
    if 'Температура' in str(listo[i]):
        temp_idx=i
    if 'Давление' in str(listo[i]):
        pressure_idx=i
    if 'Ветер' in str(listo[i]):
        wind_idx=i
    if 'Влажность' in str(listo[i]):
        hum_idx=i

temp=[]
for i in listo[temp_idx]: #индекс listo - номер строки из таблицы прогноза с rp5 с соответвующим ему параметром
    temp.append(re.split(r'\s',i)[1])
pressure=[]
#print(listo[pressure_idx])
for i in listo[pressure_idx]:
    pressure.append(re.split(r'\s',i)[1])
wind=[]
for i in listo[wind_idx]:
    if len(re.split(r'\s',i))==1:
        j=0
    else:
        j=1
    wind.append(re.split(r'\s',i)[j])
    
temperature=pd.Series(temp[1:37])
pressureHPA=pd.Series(pressure[1:37]).astype(float)*1.33322
pressure=pd.Series(pressure[1:37])
wind=pd.Series(wind[1:37])
clouds=pd.Series(clouds)
humidity=pd.Series(listo[hum_idx][1:37])
date=pd.Series(date)
day=pd.Series(day)
lat=56.0827338 
inc=sun.inc_a(lat,day,timeh,40,1)[1]
zen=sun.zen(lat,day,timeh)[0]
forecastRP5=pd.concat([date,inc,temperature,clouds,pressureHPA,humidity,wind,zen,pressure],axis=1).dropna()
forecastRP5.rename(columns={forecastRP5.columns[0]:'Date',forecastRP5.columns[1]:'inc',forecastRP5.columns[2]:'T',
                            forecastRP5.columns[3]:'Cl',forecastRP5.columns[4]:'P0',forecastRP5.columns[5]:'U',
                            forecastRP5.columns[6]:'Ff',forecastRP5.columns[7]:'zen',forecastRP5.columns[8]:'P0mm'},inplace = True)
#forecastRP5['Date']=pd.to_datetime(forecastRP5['Date'])
forecastRP5['Date']=pd.to_datetime(pd.date_range(start=today_modified, periods=36, freq='H'))
forecastRP5=forecastRP5.set_index('Date')

print(forecastRP5)