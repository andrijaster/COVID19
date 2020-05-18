import scipy.integrate as spint
from scipy import optimize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import pandas as pd
import warnings
import math
import pickle
import os
import xlwt as xl

#warnings.filterwarnings("ignore")


class SIRM_deterministic():

    def __init__(self, name = "det_T", no_countries = 20):
        self.name = name
        self.no_countries = no_countries

    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        near = idx
        if array[idx] > value:
            near = idx-1
        return near

    @staticmethod
    def solve(var, country_atribute,list_SIRM,list_inter, inter_dates):

        y = list_SIRM.values
        inter = list_inter
        inter_feature = inter.iloc[:,2:].values
        date_inter = inter.iloc[:,1].values
        I0 = y[0,1]
        S0 = y[0,0]
        R0 = y[0,2]
        M0 = y[0,3]

        t = np.arange(0,y.shape[0]-0.9, 0.1)
        time = np.linspace(0,y.shape[0]-1, y.shape[0])
        index = np.where(np.in1d(t, time))[0]

        inter_param = var[:-3]
        intercept_beta = var[-3]
        intercept_gamma = var[-2]
        intercept_mort = var[-1]

        country_beta = 0
        gamma = np.abs(intercept_gamma)
        mort = np.abs(intercept_mort)

        if gamma <10 and mort<10:
            solution = spint.odeint(SIRM_deterministic.SIRM_model, [S0, I0, R0, M0], t, 
                    args = (country_beta, gamma, mort, inter_param, inter_feature, date_inter, intercept_beta, inter_dates))
        else:
            solution = np.ones([t.shape[0],4])*100

        return np.array(solution), solution[index,:], list_SIRM

    @staticmethod 
    def SIRM_model(y, t, country_beta, gamma, mort, inter_param, inter_feature, date_inter, intercept, inter_dates):
        idx = SIRM_deterministic.find_nearest(date_inter,t)
        beta = np.abs(country_beta + (inter_feature[idx,:]*(t-inter_dates)).dot(inter_param) + intercept)
        S, I, R, M = y
        dS_dt = - beta*S*I
        dI_dt = beta*S*I - gamma*I - mort*I
        dR_dt = gamma*I
        dM_dt = mort*I
        return ([dS_dt, dI_dt, dR_dt, dM_dt])


    def fit(self, country_atribute, list_SIRM, list_inter, inter_dates, iter_max = 350, pop_size = 600, opt = 'meta'):


        def Obj_function(var, country_atribute, list_SIRM, list_inter, inter_dates):
            obj_fun = 0
            _, solution_index_list, _ = SIRM_deterministic.solve(var, country_atribute,list_SIRM,list_inter, inter_dates)
            y = list_SIRM.values
            solution = solution_index_list
            obj_fun += np.sum((solution - y)**2)
            return obj_fun

        bnd = ((-1,1),)*list_inter.iloc[:,2:].shape[1] + ((0, 5),)*3        
        
        if opt == 'meta':
            res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = iter_max, popsize = pop_size, 
                                                        args = (country_atribute, list_SIRM, list_inter, inter_dates), disp = True, tol = 0.00001)
            np.save("models_save/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        elif opt == 'neld':
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter, inter_dates),
                                                    method = 'Nelder-Mead', options={'maxiter': iter_max})
            np.save("models_save/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        elif opt == "CG":
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter, inter_dates),
                                                    method = 'CG', options={'maxiter': iter_max})
            np.save("models_save/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        
        self.res = res

    def predict(self, list_SIRM, list_inter, inter_dates, country_atribute):

        self.solution_list, self.solution_index_list, self.list_SIRM = SIRM_deterministic.solve(self.res.x, country_atribute, list_SIRM, list_inter, inter_dates)

    def evaluate(self):

        wb = xl.Workbook()
        ws1 = wb.add_sheet("R2")
        ws2 = wb.add_sheet("MSE")
        ws1_columns = population.iloc[:,0].unique()

        k = 0
        for i in ws1_columns: 
            ws1.row(0).write(k,i)
            ws2.row(0).write(k,i)
            k+=1

        res_R2 = []
        res_MSE = []

        y_true = self.list_SIRM.values
        y_predict = self.solution_index_list

        R2 = r2_score(y_true, y_predict)
        MSE = mean_squared_error(y_true, y_predict)

        res_R2.append(R2)
        res_MSE.append(MSE)

        ws1.row(1).write(0, R2)  
        ws2.row(1).write(0, MSE)    

        self.res_R2 = res_R2
        self.res_MSE = res_MSE  

        wb.save("results/results_{}.xls".format(self.name))

        return self.res_R2, self.res_MSE
    
    def plot(self, i, pop, no_countries = 20):

            
        y = self.list_SIRM
        solution = self.solution_list
        df = self.list_SIRM.values

        t = np.arange(0,y.shape[0]-0.9, 0.1)
        time = np.linspace(0,y.shape[0]-1, y.shape[0])


        plt.figure(num = i, figsize=(13,9))

        plt.plot(t, solution[:,1]*pop, label = "I(t)")
        plt.plot(t, solution[:,2]*pop, label = "R(t)")
        plt.plot(t, solution[:,3]*pop, label = "M(t)")
    
        plt.plot(time, df[:,1]*pop,'o', label = "real I(t)")
        plt.plot(time, df[:,2]*pop,'o',label = "real R(t)")
        plt.plot(time, df[:,3]*pop,'o', label = "real M(t)")

        plt.grid()
        plt.legend()
        plt.xlabel("Time [day]")
        plt.ylabel("Population No.")
        plt.title("{}".format(self.name))

        plt.savefig('images/country_{}_{}.png'.format(population.iloc[i,0], self.name))        

    def save(self, path):
        with open(path, 'wb') as f:        
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)   
 


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
                
        objekat = SIRM_deterministic(name = list_inter_x.iloc[0,0])
        # print(k,objekat.name)
        objekat.fit(CA, list_SIRM_train_x, list_inter_x, inter_dates_x, iter_max=100,pop_size=150)
        objekat.predict(list_SIRM_test_x, list_inter_x, inter_dates_x, CA)
        objekat.evaluate()
        objekat.plot(i, pop)
        objekat.save('models_pickle/model_{}.pkl'.format(objekat.name))
        k+=1







                    







   



