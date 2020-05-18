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
#warnings.filterwarnings("ignore")


class SIR_M_optimization():

    def __init__(self, name = "neT", no_countries = 20):
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
        
        solution_list = []
        solution_index_list = []
        for i in range(country_atribute.shape[0]):
            y = list_SIRM[i].values
            inter = list_inter[i]
            inter_feature = inter.iloc[:,2:].values
            date_inter = inter.iloc[:,1].values
            I0 = y[0,1]
            S0 = y[0,0]
            R0 = y[0,2]
            M0 = y[0,3]

            t = np.arange(0,y.shape[0]-0.9, 0.1)
            time = np.linspace(0,y.shape[0]-1, y.shape[0])
            index = np.where(np.in1d(t, time))[0]

            beta_param = var[:country_atribute.shape[1]]
            gama_param = var[country_atribute.shape[1]:2*country_atribute.shape[1]]
            mort_param = var[2*country_atribute.shape[1]:3*country_atribute.shape[1]]
            inter_param = var[3*country_atribute.shape[1]:-3]
            intercept_beta = var[-3]
            intercept_gamma = var[-2]
            intercept_mort = var[-1]

            country_beta = country_atribute[i,:].dot(beta_param)
            gamma = np.abs(country_atribute[i,:].dot(gama_param) + intercept_gamma)
            mort = np.abs(country_atribute[i,:].dot(mort_param) + intercept_mort)

            if gamma <10 and mort<10:
                solution = spint.odeint(SIR_M_optimization.SIRM_model, [S0, I0, R0, M0], t, 
                        args = (country_beta, gamma, mort, inter_param, inter_feature, date_inter, intercept_beta, inter_dates[i,:]))
            else:
                solution = np.ones([t.shape[0],4])*100
            solution_list.append(np.array(solution))
            solution_index_list.append(solution[index,:])

        return solution_list, solution_index_list, list_SIRM

    @staticmethod 
    def SIRM_model(y, t, country_beta, gamma, mort, inter_param, inter_feature, date_inter, intercept, inter_dates):
        idx = SIR_M_optimization.find_nearest(date_inter,t)
        beta = np.abs(country_beta + (inter_feature[idx,:]*(t-inter_dates)).dot(inter_param) + intercept)
        S, I, R, M = y
        dS_dt = - beta*S*I
        dI_dt = beta*S*I - gamma*I - mort*I
        dR_dt = gamma*I
        dM_dt = mort*I
        return ([dS_dt, dI_dt, dR_dt, dM_dt])


    def fit(self, iter_max = 350, pop_size = 600, opt = 'meta', name_SIRM_train = "dataset/podaci_SIRM.pkl", 
                name_inter_train = "dataset/podaci_INTERVENTION.pkl", 
                name_inter_dates_train = 'dataset/podaci_INTERVENTION_DATES.pkl',
                name_country_atribute = "dataset/country_atribute.csv"):


        def Obj_function(var, country_atribute, list_SIRM, list_inter, inter_dates):
            obj_fun = 0
            _, solution_index_list, _ = SIR_M_optimization.solve(var, country_atribute,list_SIRM,list_inter, inter_dates)
            for i in range(country_atribute.shape[0]):
                y = list_SIRM[i].values
                solution = solution_index_list[i]
                obj_fun += np.sum((solution - y)**2)
            return obj_fun

        with open(name_SIRM_train, 'rb') as f1:
            list_SIRM = pickle.load(f1)

        with open(name_inter_train, 'rb') as f2:
            list_inter = pickle.load(f2)   

        with open(name_inter_dates_train, 'rb') as f3:
            inter_dates = pickle.load(f3)   
        
        inter_dates = inter_dates.values
        country_atribute = pd.read_csv(name_country_atribute, index_col=0)
        bnd = ((-1,1),)*(3*country_atribute.iloc[:,2:].shape[1]) + ((-1,1),)*list_inter[0].iloc[:,2:].shape[1] + ((0, 2),)*3
        country_atribute = country_atribute.iloc[:,2:].values
        
        if opt == 'meta':
            res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = iter_max, popsize = pop_size, 
                                                        args = (country_atribute, list_SIRM, list_inter, inter_dates), disp = True, tol = 0.00001)
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        elif opt == 'neld':
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter, inter_dates),
                                                    method = 'Nelder-Mead', options={'maxiter': iter_max})
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        elif opt == "CG":
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter, inter_dates),
                                                    method = 'CG', options={'maxiter': iter_max})
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, self.name),res.x)
        
        self.res = res

    def predict(self, name_SIRM_test = "dataset/podaci_SIRM.pkl", 
                name_inter_test = "dataset/podaci_INTERVENTION.pkl", 
                name_inter_dates_test = 'dataset/podaci_INTERVENTION_DATES.pkl',
                name_country_atribute = "dataset/country_atribute.csv"):

        with open(name_SIRM_test, 'rb') as f1:
            list_SIRM = pickle.load(f1)

        with open(name_inter_test, 'rb') as f2:
            list_inter = pickle.load(f2)   
            
        with open(name_inter_dates_test, 'rb') as f3:
            inter_dates = pickle.load(f3)   

        country_atribute = pd.read_csv(name_country_atribute, index_col=0)
        country_atribute = country_atribute.iloc[:,2:].values
        inter_dates = inter_dates.values

        self.solution_list, self.solution_index_list, self.list_SIRM = SIR_M_optimization.solve(self.res.x, country_atribute, list_SIRM, list_inter, inter_dates)

    def evaluate(self):

        res_R2 = []
        res_MSE = []
        for i in range(self.no_countries):
            y_true = self.list_SIRM[i].values
            y_predict = self.solution_index_list[i]
            res_R2.append(r2_score(y_true, y_predict))
            res_MSE.append(mean_squared_error(y_true, y_predict))
        return res_R2, res_MSE
    
    def plot(self, no_countries = 20):

        population = pd.read_excel("data/Population.xlsx")

        for i in range(self.no_countries):
            
            y = self.list_SIRM[i]
            solution = self.solution_list[i]
            pop = population.iloc[i,1]   
            df = self.list_SIRM[i]

            t = np.arange(0,y.shape[0]-0.9, 0.1)
            time = np.linspace(0,y.shape[0]-1, y.shape[0])

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
            plt.title("{}".population[i,0])

            plt.savefig('images/country_{}.png'.population[i,0])

    def save(self, path):
        with open(path, 'w') as f:        
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'r') as f:
            self.__dict__.update(pickle.load(f).__dict__)    


def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

if __name__=="__main__":
    file_path = "models_pickle"
    directory = ensure_dir(file_path)

    objekat = SIR_M_optimization()
    objekat.fit(iter_max=1,pop_size=1)
    objekat.predict()
    objekat.evaluate()
    objekat.plot()

    objekat.save('{}/model_{}.pkl'.format(file_path, objekat.name))




                    







   



