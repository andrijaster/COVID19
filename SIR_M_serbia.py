# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:22:10 2020

@author: Andri
"""
import scipy.integrate as spint
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt 
import csv


def Obj_function(var, t, df, ind, I0, R0, M0, S0):
    beta = var[0]
    gamma = var[1]
    mort = var[2]
    solution = spint.odeint(SIR_model, [I0, R0, M0, S0], t, args= (beta, gamma, mort))
    solution = np.array(solution)
    fun_rel = solution[ind,:]
    obj_fun = np.sum((fun_rel - df)**2)
    return obj_fun

def Obj_function_2(var, sol, domen_1, df):
    fun_rel = model_new(var,sol, domen_1)
    obj_fun = np.sum((fun_rel - df)**2)
    return obj_fun

def model_new(var, sol, domen_1):
    var_sum = 0
    model = 0
    for i in range(domen_1.shape[0]):
        model += var[i]*sol[i]
        var_sum += var[i]  
    fun_rel = model/var_sum
    return fun_rel
            
def SIR_model_2(y, t, beta, gamma, mort, prekid):
    if t > prekid:
        beta = beta[1]
    else:
        beta = beta[0]
    I, R, M, S = y
    dS_dt = - beta*S*I
    dS_dt = - beta*S*I
    dI_dt = beta*S*I - gamma*I - mort*I
    dR_dt = gamma*I
    dM_dt = mort*I
    return ([dI_dt, dR_dt, dM_dt, dS_dt])

def SIR_model(y, t, beta, gamma, mort):
    I, R, M, S = y
    dS_dt = - beta*S*I
    dS_dt = - beta*S*I
    dI_dt = beta*S*I - gamma*I - mort*I
    dR_dt = gamma*I
    dM_dt = mort*I
    return ([dI_dt, dR_dt, dM_dt, dS_dt])

if __name__ == "__main__":
    opt = 'meta'
    fit = 0 # Nauci parametar sirenja infekcije za srbiju
    fit_2 = 0
    country = 'Serbia'
    pop = 6982604 #Serbia
    domen_1 = np.arange(27,40) # dani od kada se pretpostavlja da ce krenuti uticaj karantina
    
    
    name = "dataset/df_{}.npy".format(country)
    df = np.load(name, allow_pickle = True)
    df = df[:,:]

    I0 = df[0,0]
    R0 = df[0,1]
    M0 = df[0,2]
    S0 = df[0,3]
    
    t = np.linspace(0,50,501)
    time = np.arange(0,df.shape[0])
    index = np.where(np.in1d(t, time))[0]
    bnd = ((0,6),)*3
    
    if fit == 1:
        res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = 200, popsize = 1000,
                            args = (t, df, index, I0, R0, M0, S0), disp = True, tol = 0.0000001)
        np.save("modeli/vektor_{}".format(country),res.x)

                
        

        
    sol = []
    beta = np.zeros(2)
    
    
    for prekid in domen_1:
        res_kine = np.load('modeli/vektor_{}_{}.npy'.format(prekid-9,'China'))
        res_srbije = np.load("modeli/vektor_{}.npy".format(country))
        
        beta[0] = res_srbije[0]
        beta[1] = res_kine[1]
        gamma = res_kine[2]
        mort = res_kine[3]
    
        solution = spint.odeint(SIR_model_2, [I0, R0, M0, S0], t, args= (beta, gamma, mort, prekid))
        sol.append(np.array(solution))
        
        plt.figure(figsize = [10,6])
        plt.plot(t, solution[:,0]*pop, label = "I(t)")
        plt.plot(t, solution[:,1]*pop, label = "R(t)")
        plt.plot(t, solution[:,2]*pop, label = "M(t)")
        
        plt.plot(time, df[:,0]*pop,'o', label = "real I(t)")
        plt.plot(time, df[:,1]*pop,'o',label = "real R(t)")
        plt.plot(time, df[:,2]*pop,'o', label = "real M(t)")
        
        plt.grid()
        plt.legend()
        plt.xlabel("Vreme")
        plt.ylabel("Broj_stanovnika")
        plt.show()
        
        plt.savefig('images/slika_{}_{}.png'.format(prekid,country))
        
    """ IZRACUNAJ SA OSREDNJENIM KRIVAMA """
    solution = np.mean(sol,axis = 0)
    plt.figure(figsize = [10,6])
    
    plt.plot(t, solution[:,0]*pop, label = "I(t)")
    plt.plot(t, solution[:,1]*pop, label = "R(t)")
    plt.plot(t, solution[:,2]*pop, label = "M(t)")
    
    plt.plot(time, df[:,0]*pop,'o', label = "real I(t)")
    plt.plot(time, df[:,1]*pop,'o',label = "real R(t)")
    plt.plot(time, df[:,2]*pop,'o', label = "real M(t)")
    
    plt.grid()
    plt.legend()
    plt.xlabel("Vreme")
    plt.ylabel("Broj ljudi")
    plt.show()
    
    plt.savefig('images/slika_{}_{}.png'.format('mean',country))       
    
    """ NAUCI TEZINE ZA SVAKU ZEMLJU """
    
    if fit_2 == 1:
        res = []
        for i in range(len(domen_1)):
            res.append(sol[i][index])
        
        bounds =  bnd = ((0,10),)*len(domen_1)
        res_2 = optimize.differential_evolution(Obj_function_2, bounds = bnd, maxiter = 300, popsize = 1000,
                                            args = (res, domen_1, df), disp = True, tol = 0.0000001)
        np.save('modeli/tezine_SRB', res_2.x)
        var = np.load("modeli/tezine_SRB.npy") 
    else:
        var = np.load("modeli/tezine.npy") 
    
    solution_2 = model_new(var, sol, domen_1)
    
    # solution = np.mean(sol,axis = 0)
    plt.figure(figsize = [10,6])
    
    plt.plot(t, solution_2[:,0]*pop, label = "I(t)")
    plt.plot(t, solution_2[:,1]*pop, label = "R(t)")
    plt.plot(t, solution_2[:,2]*pop, label = "M(t)")
    
    plt.plot(time, df[:,0]*pop,'o', label = "real I(t)")
    plt.plot(time, df[:,1]*pop,'o',label = "real R(t)")
    plt.plot(time, df[:,2]*pop,'o', label = "real M(t)")
    
    plt.grid()
    plt.legend()
    plt.xlabel("Vreme")
    plt.ylabel("Broj ljudi")
    plt.show()
    
    plt.savefig('images/slika_{}_{}.png'.format('mean_w',country))
    
    datum = np.arange('2020-03-06', '2020-04-26', dtype='datetime64[D]')
    time = np.arange(0,51)
    index = np.where(np.in1d(t, time))[0]
    table = solution[index,:-1]*pop
    # table = np.insert(table,0,datum,axis=1)
    
    with open('text.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(datum,table))
    
    