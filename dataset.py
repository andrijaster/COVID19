import pandas as pd
import numpy as np

data_infected = pd.read_csv("data/time_series_19-covid-Confirmed.csv")
data_recovered = pd.read_csv("data/time_series_19-covid-Recovered.csv")
data_death = pd.read_csv("data/time_series_19-covid-Deaths.csv")

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

country = 'Serbia'
data_serbia_infected = data_infected[country]
data_serbia_recovered = data_recovered[country]
data_serbia_death = data_death[country]



d = {'Infected' : data_infected, 'Recovered' : data_recovered, 'Death': data_death}

data = pd.concat(d.values(), axis=1, keys=d.keys())

mort = data_death.sum(axis=1)/data_infected.sum(axis=1)

mort_tot = data_death.sum(axis=1).sum()/data_infected.sum(axis = 1).sum()
mort_tot_2 = mort.mean()

gamma = data_recovered.sum(axis=1)/data_infected.sum(axis=1)


gamma_tot = data_recovered.sum(axis=1).sum()/data_infected.sum(axis=1).sum()
gamma_tot_2 = gamma.mean()

# pop_serbia = 6982604
# pop_italy = 10078012
pop_china_hubeid = 58500000

pop = 6982604

country = 'Serbia'


lista = [('Infected',country),('Recovered',country),('Death',country)]
df = data.loc[:,lista]
if country == 'China':
    uslov = df.iloc[0,:] == 'Hubei'
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


