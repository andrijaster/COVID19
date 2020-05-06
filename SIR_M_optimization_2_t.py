import scipy.integrate as spint
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import pandas as pd


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    near = idx
    if array[idx] > value:
        near = idx-1
    return near

def Obj_function(var, country_atribute, list_SIRM, list_inter):

    obj_fun = 0
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
        inter_param = var[3*country_atribute.shape[1]:-5]
        intercept_beta = var[-5]
        intercept_gamma = var[-4]
        intercept_mort = var[-3]
        beta_time_param = var[-2:]
        
        country_beta = country_atribute[i,:].dot(beta_param)
        gamma = np.exp(country_atribute[i,:].dot(gama_param) + intercept_gamma)
        mort = np.exp(country_atribute[i,:].dot(mort_param) + intercept_mort)


        solution = spint.odeint(SIRM_model, [S0, I0, R0, M0], t, 
                    args = (country_beta, gamma, mort, inter_param, inter_feature, date_inter, 
                            intercept_beta, beta_time_param,))
        solution = np.array(solution)
        obj_fun += np.sum((solution[index,:] - y)**2)
    return obj_fun

        
def SIRM_model(y, t, country_beta, gamma, mort, inter_param, inter_feature, date_inter, intercept, beta_time_param):
    time = np.array([t , t**2])
    if t > date_inter[0]:
        idx = find_nearest(date_inter,t)
        beta = np.exp(country_beta + inter_feature[idx,:].dot(inter_param) + intercept + beta_time_param.dot(time))
    else:
        beta = np.exp(country_beta + intercept + beta_time_param.dot(time))
    S, I, R, M = y
    dS_dt = - beta*S*I
    dI_dt = beta*S*I - gamma*I - mort*I
    dR_dt = gamma*I
    dM_dt = mort*I
    return ([dS_dt, dI_dt, dR_dt, dM_dt])



if __name__ == "__main__":
    name = "daT"
    opt = 'meta'
    fit = 1
    iter_max = 1
    pop_size = 1

    
    with open('dataset/podaci_SIRM.pkl', 'rb') as f1:
        list_SIRM = pickle.load(f1)

    with open('dataset/podaci_INTERVENTION.pkl', 'rb') as f2:
        list_inter = pickle.load(f2)   

    country_atribute = pd.read_csv("dataset/country_atribute.csv", index_col=0)
    
    
    bnd = ((-1,1),)*(3*country_atribute.iloc[:,2:].shape[1]) + ((-1,0),)*list_inter[0].iloc[:,2:].shape[1] + ((0, 2),)*3 + ((-3,3),)*2
    country_atribute = country_atribute.iloc[:,2:].values

    
    if fit == 1:
        if opt == 'meta':
            res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = iter_max, popsize = pop_size, 
            args = (country_atribute, list_SIRM, list_inter,), disp = True, tol = 0.00001)
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, name),res.x)
        elif opt == 'neld':
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter,),
                                                    method = 'Nelder-Mead', options={'maxiter': iter_max})
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, name),res.x)
        elif opt == "CG":
            x0 = np.zeros(len(bnd))
            res = optimize.minimize(Obj_function, x0,  args = (country_atribute, list_SIRM, list_inter,),
                                                    method = 'CG', options={'maxiter': iter_max})
            np.save("modeli/vektor_{}_{}_{}_{}".format(iter_max, pop_size, opt, name),res.x)