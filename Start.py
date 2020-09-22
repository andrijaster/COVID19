import os
import pickle 
from models import SIRM_deterministic
from models import SIRM_deterministic_T2
from models import SIRM_deterministic_per_country
from models import SIRM_deterministic_per_countr_date_1
from models import SIRM_deterministic_per_countr_date_1_T2
from models import SIRM_deterministic_per_countr_date_1_ver_1
from models import SIRM_deterministic_per_countr_date_1_T2_ver_1
import pandas as pd

if __name__=="__main__":
    
    def ensure_dir(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    name_SIRM_train = "dataset/podaci_SIRM_train.pkl"
    name_SIRM_test = "dataset/podaci_SIRM_test.pkl"
    name_inter_test = "dataset/podaci_INTERVENTION.pkl"
    name_inter_dates_test = 'dataset/podaci_INTERVENTION_DATES.pkl'
    name_country_atribute = "dataset/country_atribute.csv"


    with open(name_SIRM_train, 'rb') as f1:
        list_SIRM_train = pickle.load(f1)

    with open(name_SIRM_test, 'rb') as f1:
        list_SIRM_test = pickle.load(f1)

    with open(name_inter_test, 'rb') as f2:
        list_inter = pickle.load(f2)   
        
    with open(name_inter_dates_test, 'rb') as f3:
        inter_dates = pickle.load(f3)   

    population = pd.read_excel("data/Population.xlsx")

    country_atribute = pd.read_csv(name_country_atribute, index_col=0)
    country_atribute = country_atribute.iloc[:,2:].values
    inter_dates = inter_dates.values

    file_path = "models_pickle"
    ensure_dir(file_path)

    k=0
    for i in range(country_atribute.shape[0]):
        
        CA = country_atribute[i]
        list_inter_x = list_inter[i]
        list_SIRM_test_x = list_SIRM_test[i]
        list_SIRM_train_x = list_SIRM_train[i]
        inter_dates_x =  inter_dates[i]
        pop = population.iloc[i,1] 
                
        objekat = SIRM_deterministic_per_countr_date_1_T2(name = list_inter_x.iloc[0,0])
        objekat.fit(CA, list_SIRM_train_x, list_inter_x, inter_dates_x, iter_max=500 ,pop_size=500)
        objekat.predict(list_SIRM_test_x, list_inter_x, inter_dates_x, CA)
        objekat.evaluate(population)
        objekat.plot(i, pop, population)
        objekat.save('models_pickle/model_{}.pkl'.format(objekat.name))
        k+=1

