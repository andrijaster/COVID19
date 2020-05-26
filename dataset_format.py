import pandas as pd
import numpy as np
import pickle

from datetime import timedelta
from sklearn import preprocessing

data_infected = pd.read_csv("data/time_series_covid19_confirmed_global.csv")
data_recovered = pd.read_csv("data/time_series_covid19_recovered_global.csv")
data_death = pd.read_csv("data/time_series_covid19_deaths_global.csv")

hel_nut_pop = pd.read_csv("data/health_nutrition_and_population_2019.csv")
bussines = pd.read_csv("data/doing_business_2019.csv")
population = pd.read_excel("data/Population.xlsx")

# country_atribute = hel_nut_pop.join(bussines.set_index('Economy Name'),how = 'inner', on = 'Country Name')

country_atribute = hel_nut_pop

country = ["China", "Italy", "Spain", "Germany", "Iran", "France", "Switzerland", 
"UK", "Norway", "Serbia", "Greece", "Hungary", "Belgium", "Turkey", 
"Netherlands", "Portugal", "Sweden","Croatia", "Belarus", "Ireland"]

country_2 = ["China", "Italy", "Spain", "Germany", "Iran", "France", "Switzerland", 
"United Kingdom", "Norway", "Serbia", "Greece", "Hungary", "Belgium", "Turkey", 
"Netherlands", "Portugal", "Sweden","Croatia", "Belarus", "Ireland"]

country_atribute = country_atribute.loc[country_atribute["Country Name"].isin(country)]
country_atribute.reset_index(inplace = True, drop=True)

intervention = pd.read_excel("data/acaps_covid19_goverment_measures_dataset.xlsx",sheet_name="Database")
kolone = ["COUNTRY", "MEASURE", "TARGETED_POP_GROUP","DATE_IMPLEMENTED","SOURCE_TYPE"]
intervention = intervention[kolone]
intervention = intervention.loc[intervention["TARGETED_POP_GROUP"] == "No"]
intervention = intervention.loc[intervention["SOURCE_TYPE"] == "Government"]
intervention = intervention.loc[intervention["COUNTRY"].isin(country_2)]

intr = pd.get_dummies(intervention["MEASURE"])
intr = intervention.join(intr)
kolone = ["MEASURE","TARGETED_POP_GROUP","SOURCE_TYPE"]
intr.drop(kolone, axis = 1, inplace = True)
intr = intr.dropna()

# ----- DODATO -----
intr_2 = intervention.copy()
intr_2 = intr_2.groupby(['COUNTRY', 'MEASURE'])['DATE_IMPLEMENTED'].min().reset_index()
#intr_2 = intr_2.drop_duplicates(subset=['DATE_IMPLEMENTED'], keep = "last")
#dates_of_interventions = intr_2.groupby('COUNTRY').apply(lambda x: x['DATE_IMPLEMENTED']).reset_index()

value_vars = data_infected.columns[4:]
first_occ = []
for c in intr_2['COUNTRY'].unique():

    if c == 'China':
        i = data_infected.loc[(data_infected['Country/Region'] == c) & (data_infected['Province/State'] == 'Hubei'), value_vars]
    else:
        i = data_infected.loc[data_infected['Country/Region'] == c, value_vars]
    i = i.sum()
    inx = np.min(np.where(i > 0))
    first_occ.append([c, data_infected.columns[inx + 4]])

first_occ = pd.DataFrame(first_occ, columns=['COUNTRY', 'DATE_FIRST'])
intr_2 = intr_2.merge(first_occ, on='COUNTRY')
intr_2['DATE_FIRST'] = pd.to_datetime(intr_2['DATE_FIRST'])
intr_2['DAYS_FROM_FIRST'] = (intr_2['DATE_IMPLEMENTED'] - intr_2['DATE_FIRST'])/np.timedelta64(1, 'D')

# AKO SU MERE DODATE PRE PRVOG POJAVLJIVANJA STAVITI NULA
intr_2.loc[intr_2['DAYS_FROM_FIRST'] < 0, 'DAYS_FROM_FIRST'] = 0

pivot_interventions = intr_2.groupby(['COUNTRY', 'MEASURE'])['DAYS_FROM_FIRST'].min().reset_index()
pivot_interventions = pivot_interventions.pivot(index='COUNTRY', columns='MEASURE', values='DAYS_FROM_FIRST').fillna(0)

# ----- END DODATO -----

list_intervention = []
for i in intr['COUNTRY'].unique():
    intr_1 = intr[intr['COUNTRY']==i]
    intr_1 = intr_1.sort_values(by='DATE_IMPLEMENTED')
    medju = intr_1[intr_1['COUNTRY']==i].iloc[:,2:].cumsum().values
    medju[medju>1] = 1
    intr_1.iloc[:,2:] = medju
    kolone = intr_1.columns[2:]
    intr_1.reset_index(inplace = True, drop = True)
    intr_1 = intr_1.drop_duplicates(subset=kolone)
    if i == "United Kingdom":
        intr_1 = intr_1.replace(to_replace = "United Kingdom", value = "UK")
    list_intervention.append(intr_1)


intervention = intr.groupby(["COUNTRY"]).sum()
intervention[intervention>1]=1
intervention = intervention.T.rename(columns = {"United Kingdom": "UK"}).T 

lista_SIRM_trening = []
lista_SIRM_test = []
lista_SIR = []

country_atribute["Lat"] = np.zeros(country_atribute.shape[0])
country_atribute["Long"] = np.zeros(country_atribute.shape[0])

k = 0

country = intervention.index

for i in country:
    if i == "China":
        i = "Hubei"
        kolona = 0
    else:
        kolona = 1
    df_rec = data_recovered[data_recovered.iloc[:,kolona] == i].iloc[:,1:]
    df_death = data_death[data_death.iloc[:,kolona] == i].iloc[:,1:]
    df_infected = data_infected[data_infected.iloc[:,kolona] == i].iloc[:,1:]
    
    df_rec = df_rec.iloc[:,4:]
    df_infected = df_infected.iloc[:,4:]
    df_death = df_death.iloc[:,4:]

    df_infected.iloc[0,:] = df_infected.iloc[0,:] - df_rec.iloc[0,:] - df_death.iloc[0,:]
    df_infected = df_infected.T[df_infected.any()].T.dropna(axis = 1) 
    kolone = df_infected.columns

    df_rec = df_rec.loc[:,kolone]/population.iloc[k,1]
    df_death = df_death.loc[:,kolone]/population.iloc[k,1]
    df_infected = df_infected/population.iloc[k,1]

    


    df_rec_new = df_rec.T.reset_index()
    df_rec_new = pd.to_datetime(df_rec_new.iloc[:,0])

    intr_1 = list_intervention[k]
    intr_1['DATE_IMPLEMENTED'] = intr_1['DATE_IMPLEMENTED']
    intr_1 = intr_1[intr_1['DATE_IMPLEMENTED']<df_rec_new.iloc[-1]]
    

    
    for j in range(intr_1.shape[0]):
        mesto = df_rec_new == intr_1.iloc[j,1]
        if mesto.any():
            intr_1.iloc[j,1] = df_rec_new[mesto].index
        else: 
            intr_1.iloc[j,1] = 0

    intr_1.loc[-1] = 0
    intr_1.index = intr_1.index + 1
    intr_1.sort_index(inplace=True)
    intr_1.iloc[0,0] = intr_1.iloc[1,0]
    intr_1 = intr_1.drop_duplicates(subset=["DATE_IMPLEMENTED"], keep = "last")

    list_intervention[k] = intr_1

    df_rec = df_rec.T.reset_index(drop = True)
    df_death = df_death.T.reset_index(drop = True)
    df_infected = df_infected.T.reset_index(drop = True)
    df_s = 1 - df_rec - df_death.values - df_infected.values

    """ SIRM """
    new_df = pd.concat([df_s, df_infected, df_rec, df_death], axis = 1)   
    kolone_imena = ["S", "Infected", "Recovered", "Death"]
    new_df = new_df.T.reset_index(drop=True).T

    dicts = {}
    ind = 0
    for values in kolone_imena:
        new_df.rename(columns={new_df.columns[ind]: values}, inplace = True)
        ind+=1

    trening_NO = int(new_df.shape[0]*0.8)

    lista_SIRM_trening.append(new_df.iloc[0:trening_NO,:])
    lista_SIRM_test.append(new_df)

    """ SIR """
    df_recovered = df_rec + df_death.values
    new_df_SIR = pd.concat([df_s, df_infected, df_recovered], axis = 1)
    kolone_imena = ["S", "Infected", "Recovered+Death"]
    new_df_SIR = new_df_SIR.T.reset_index(drop=True).T

    dicts = {}
    ind = 0
    for values in kolone_imena:
        new_df_SIR.rename(columns={new_df_SIR.columns[ind]: values}, inplace = True)
        ind+=1

    lista_SIR.append(new_df_SIR)
    k+=1


with open('dataset/podaci_SIRM_train.pkl', 'wb') as f1:
    pickle.dump(lista_SIRM_trening, f1)

with open('dataset/podaci_SIRM_test.pkl', 'wb') as f1:
    pickle.dump(lista_SIRM_test, f1)
    
with open('dataset/podaci_SIR.pkl', 'wb') as f2:
    pickle.dump(lista_SIR, f2)

with open('dataset/podaci_INTERVENTION.pkl', 'wb') as f3:
    pickle.dump(list_intervention, f3)
    
with open('dataset/podaci_INTERVENTION_DATES.pkl', 'wb') as f4:
    pickle.dump(pivot_interventions, f4)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    near = idx
    if array[idx] > value:
        near = idx-1
    return near


list_intervention_NN = []
for i in range(len(lista_SIRM_test)):
    tabela = np.zeros([lista_SIRM_test[i].shape[0],list_intervention[i].iloc[:,2:].shape[1]])
    for j in range(tabela.shape[0]):
        array = list_intervention[i].iloc[:,1].values
        idx = find_nearest(array,j)
        tabela[j,:] = list_intervention[i].iloc[idx,2:].values.copy()
    list_intervention_NN.append(tabela)

with open('dataset/podaci_INTERVENTION_NN.pkl', 'wb') as f3:
    pickle.dump(list_intervention_NN, f3)


scale = preprocessing.StandardScaler()

country_atribute = country_atribute.dropna(axis=1)
x_scaled = scale.fit_transform(country_atribute.iloc[:,1:])
country_atribute.iloc[:,1:] = x_scaled
country_atribute.to_csv('dataset/country_atribute.csv')

for i in range(20):
    print(list_intervention[i].iloc[0,0], country_atribute.iloc[i,0])

        
