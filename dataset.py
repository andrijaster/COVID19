import pandas as pd
import numpy as np

data_infected = pd.read_csv("data/time_series_19-covid-Confirmed.csv")
data_recovered = pd.read_csv("data/time_series_19-covid-Recovered.csv")
data_death = pd.read_csv("data/time_series_19-covid-Deaths.csv")

data_infected = data_infected.iloc[:,1:].T
data_recovered = data_recovered.iloc[:,1:].T
data_death = data_death.iloc[:,1:].T

data_infected.rename(columns=data_infected.iloc[0], inplace = True)
data_infected = data_infected.iloc[3:,:]
data_recovered.rename(columns=data_recovered.iloc[0], inplace = True)
data_recovered = data_recovered.iloc[3:,:]
data_death.rename(columns=data_death.iloc[0], inplace = True)
data_death = data_death.iloc[3:,:]

country = 'Serbia'
data_serbia_infected = data_infected[country]
data_serbia_recovered = data_recovered[country]
data_serbia_death = data_death[country]


d = {'Infected' : data_infected, 'Recovered' : data_recovered, 'Death': data_death}
df = [data_infected, data_recovered, data_death]
data = pd.concat(d.values(), axis=1, keys=d.keys())

mort = data_death.sum(axis=1)/data_infected.sum(axis=1)

mort_tot = data_death.sum(axis=1).sum()/data_infected.sum(axis = 1).sum()
mort_tot_2 = mort.mean()

gamma = data_recovered.sum(axis=1)/data_infected.sum(axis=1)


gamma_tot = data_recovered.sum(axis=1).sum()/data_infected.sum(axis=1).sum()
gamma_tot_2 = gamma.mean()
