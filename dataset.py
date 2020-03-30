import pandas as pd
import numpy as np

country = 'Serbia'
provincija = 'Hubei' # Upisati provinciju za Kinu

data_infected = pd.read_csv("data/time_series_covid19_confirmed_global.csv")
data_recovered = pd.read_csv("data/time_series_covid19_recovered_global.csv")
data_death = pd.read_csv("data/time_series_covid19_deaths_global.csv")

data_infected = data_infected.T
data_recovered = data_recovered.T
data_death = data_death.T

kolone = np.r_[0, -60:0]
data_infected.rename(columns=data_infected.iloc[1], inplace = True)
data_infected = data_infected.iloc[kolone,:]
data_recovered.rename(columns=data_recovered.iloc[1], inplace = True)
data_recovered = data_recovered.iloc[kolone,:]
data_death.rename(columns=data_death.iloc[1], inplace = True)
data_death = data_death.iloc[kolone,:]

data_serbia_infected = data_infected[country]
data_serbia_recovered = data_recovered[country]
data_serbia_death = data_death[country]



d = {'Infected' : data_infected, 'Recovered' : data_recovered, 'Death': data_death}
data = pd.concat(d.values(), axis=1, keys=d.keys())

# pop_serbia = 6982604
# pop_italy = 10078012
# pop_china_hubeid = 58500000

pop = 6982604

lista = [('Infected',country),('Recovered',country),('Death',country)]
df = data.loc[:,lista]
if country == 'China':
    uslov = df.iloc[0,:] == provincija
    vektor = np.where(uslov)
    df = df.iloc[1:,vektor[0]]  

df['S'] = np.zeros(df.shape[0])
df = df[df.iloc[:,0]>0]
df.iloc[:,0] = df.iloc[:,0] - df.iloc[:,1] - df.iloc[:,2]

for i in range(0,df.shape[0]):
        df.iloc[i,3] = pop - df.iloc[i,0] - df.iloc[i,1] - df.iloc[i,2]
        
df_t = df/pop

name = "dataset/df_{}".format(country)
np.save(name, df_t.iloc[:,:])


