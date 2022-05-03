import numpy as np
class sunny:
    Isc=1367 #солнечная постоянная Вт/м2
    r=np.pi/180 #множитель перевода в радианы
    deg=180/np.pi #множитель перевода в градусы
    
    def decl(self,day):
        return 23.45*np.sin((360*(day+284)/365)*self.r) #угол склонения (в градусах) по формуле Купера
    
    def w(self,h): #функция расчета часового угла движения Солнца
        midday=12 #местный астрономический полдень
        ang=15 #угловая скорость движения солнца по небосводу
        return ang*(h-midday)
    
    def alt(self,lat,day,hour):
        sin_alt=np.sin(lat*self.r)*np.sin(self.decl(day)*self.r)+np.cos(lat*self.r)*np.cos(self.decl(day)*self.r)*np.cos(self.w(hour)*self.r) #синус высоты солнца над горизонтом
        cos_alt=(1-sin_alt**2)**(0.5)
        return np.arctan(sin_alt/cos_alt)*self.deg #угол высоты Солнца над горизонтом
    
    def zen(self,lat,day,hour):
        cos_zen=np.sin(lat*self.r)*np.sin(self.decl(day)*self.r)+np.cos(self.decl(day)*self.r)*np.cos(lat*self.r)*np.cos(self.w(hour)*self.r) #косинус зенитного угла Солнца к горизонтальной площадке
        return np.arccos(cos_zen)*self.deg, cos_zen #зенитный угол
    
    def azim(self,lat,day,hour,array_case):
        hour=np.array(hour)
        day=np.array(day)
        cos_azim=(np.sin(self.alt(lat,day,hour)*self.r)*np.sin(lat*self.r)-np.sin(self.decl(day)*self.r))/(np.cos(self.alt(lat,day,hour)*self.r)*np.cos(lat*self.r))
        if array_case==0:
            if hour>0 and hour<12:
                azimuth=np.arccos(cos_azim)*self.deg*-1
            elif hour==12:
                azimuth=0
            else: 
                azimuth=np.arccos(cos_azim)*self.deg
        if array_case==1:
            indexes12=np.argwhere(hour==12) #определяем индексы элементов в массиве со значениями "12"
            indexes_dawn=np.argwhere(hour<12)
            indexes_sunset=np.argwhere(hour>12)
            np.put(cos_azim,[indexes12],0) #заменяем значения в массиве по указанным индексам
            np.put(cos_azim,[indexes_dawn],np.arccos(cos_azim[indexes_dawn])*self.deg*(-1)) #задаем отрицательный азимутальный угол до полудня
            np.put(cos_azim,[indexes_sunset],np.arccos(cos_azim[indexes_sunset])*self.deg)
            azimuth=cos_azim            
        return azimuth,np.sin(azimuth) #азимут Солнца в гр и синус азимута
    
    def sunrise_a(self,lat,day,tilting):
        return np.arccos(-np.tan(self.decl(day)*self.r)*np.tan((lat-tilting)*self.r))*self.deg #часовой угол Восхода Солнца
    
    def inc_a(self,lat,day,hour,tilting,plane_azimuth):
        cos_inc=np.sin(self.decl(day)*self.r)*np.sin(lat*self.r)*np.cos(tilting*self.r)-np.sin(self.decl(day)*self.r)*np.cos(lat*self.r)*np.sin(tilting*self.r)*np.cos(plane_azimuth*self.r)+np.cos(self.decl(day)*self.r)*np.cos(lat*self.r)*np.cos(tilting*self.r)*np.cos(self.w(hour)*self.r)+np.cos(self.decl(day)*self.r)*np.sin(lat*self.r)*np.sin(tilting*self.r)*np.cos(plane_azimuth*self.r)*np.cos(self.w(hour)*self.r)+np.cos(self.decl(day)*self.r)*np.sin(tilting*self.r)*np.sin(plane_azimuth*self.r)*np.sin(self.w(hour)*self.r)
        return np.arccos(cos_inc)*self.deg,cos_inc #угол падения Солнечной радиации

    def Iext_h(self,lat,day,hour):
        return self.Isc*(1+0.033*np.cos(day*self.r*360/365))*(np.sin(lat*self.r)*np.sin(self.decl(day)*self.r)+np.cos(lat*self.r)*np.cos(self.decl(day)*self.r)*np.cos(self.w(hour)*self.r)) #заатмосферная радиация

    def Iext_tilt(self,lat,day,hour,tilting,plane_azimuth):
        return self.Iext_h(lat,day,hour)*((self.inc_a(lat,day,hour,tilting,plane_azimuth))[1]/self.zen(lat,day,hour)[1])

    def Ics(self,lat,alt,month,day,hour):
        mr=(self.zen(lat,day,hour)[1]+0.15*(93.885-self.zen(lat,day,hour)[0])**(-1.253))**(-1) #относительная атм.масса по Кастену Икбал, 1980, формула 5.7.2
        mrR=35/(1224*self.zen(lat,day,hour)[1]+1)**0.5 #отн.атм.масса Роджерса (Икбал, 1980, формула 5.11.1
        TL=np.array([2.9,2.6,2.4,2.5,2.6,2.9,3.4,3.6,3.3,3.5,2.9,2.4,2.2],dtype='f') #Linke turbidity для Новочебоксарска (взято с soda.com) - первый элемент среднегодовое значение, остальные - по месяцам года по порядку
        fh1=np.exp(-alt/8000)
        fh2=np.exp(-alt/1250)
        cg1=(0.0000509*alt1+0.868)
        cg2=0.0000392*alt1+0.0387
        return 0.84*self.Iext_h(lat,day,hour)*self.zen(lat,day,hour)[1]*np.exp(-0.027*mr*(fh1+(TL[month]-1)*fh2)), cg1*self.Iext_h(lat,day,hour)*self.zen(lat,day,hour)[1]*np.exp(-cg2*mr*(fh1+(TL[month]-1)*fh2))*np.exp(0.01*mr**1.8) #1-модель Кастена, 2-модель Кастена с поправками Переза
    
    def I_tilt(self,lat,day,hour,tilting,plane_azimuth,H,Hb,Hd): #коэффициент пересчета с горизонтальной на наклонную
        Rb=self.inc_a(lat,day,hour,tilting,plane_azimuth)[1]/self.zen(lat,day,hour)[1]
        Rd=np.cos(0.5*tilting*self.r)**2
        if day.any()<100 and day.any()>260:
            alb=0.7
        else:
            alb=0.2
        Rr=alb*(np.sin(0.5*tilting*self.r)**2)
        return Hb*Rb+Hd*Rd+H*Rr
