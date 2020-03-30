import scipy.integrate as spint
from scipy import optimize
import numpy as np 
import matplotlib.pyplot as plt 


def Obj_function(var, t, df, ind, prekid, I0, R0, M0, S0):
    beta = var[0:2]
    gamma = var[2]
    mort = var[3]
    solution = spint.odeint(SIR_model, [I0, R0, M0, S0], t, args= (beta, gamma, mort, prekid))
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
        
def SIR_model(y, t, beta, gamma, mort, prekid):
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

if __name__ == "__main__":
    opt = 'meta'
    fit = 0 # Nauci parametre kineske
    fit_2 = 1 # Nauci tezine za svaku krivu 
    dani = 70 # broj dana od 0 kada se vrsi fitovanje ili predvidjanje
    domen = np.arange(14,31) # redni broj dana od kada deluje karantin

    country = 'China'
    pop = 58500000
    
    name = "dataset/df_{}.npy".format(country)
    df = np.load(name, allow_pickle = True)
    df = df[:,:]

    I0 = df[0,0]
    R0 = df[0,1]
    M0 = df[0,2]
    S0 = df[0,3]
    
    t = np.linspace(0,dani,dani*10+1) 
    time = np.arange(0,df.shape[0])
    index = np.where(np.in1d(t, time))[0]
    bnd = ((0,5),)*4

    
    if fit == 1:
        for prekid in domen:
            if opt == 'meta':
                res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = 300, popsize = 1000,
                                    args = (t, df, index, prekid, I0, R0, M0, S0), disp = True, tol = 0.0000001)
                np.save("modeli/vektor_{}_{}".format(prekid,country),res.x)
                beta = res.x[0:2]
                gamma = res.x[2]
                mort = res.x[3]
            else:
                x0 = np.array([0,0,0,0])
                res = optimize.minimize(Obj_function, x0,  args = (t, df, index, prekid, I0, R0, M0, S0),
                                                      method = 'Nelder-Mead')
                np.save("modeli/vektor_{}_{}".format(prekid,country),res.x)
                beta = res.x[0:2]
                gamma = res.x[2]
                mort = res.x[3]
                        
    sol = []
    domen_1 = np.arange(18,31)
    for prekid in domen_1:
        res = np.load('modeli/vektor_{}_{}.npy'.format(prekid,country))
        beta = res[0:2]
        gamma = res[2]
        mort = res[3]
        solution = spint.odeint(SIR_model, [I0, R0, M0, S0], t, args= (beta, gamma, mort, prekid))
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
    plt.ylabel("Udeli")
    plt.show()
    
    plt.savefig('images/slika_{}_{}.png'.format('mean',country))
    
    if fit_2 == 1:
        res = []
        for i in range(len(domen_1)):
            res.append(sol[i][index])
        
        bounds =  bnd = ((0,10),)*len(domen_1)
        res_2 = optimize.differential_evolution(Obj_function_2, bounds = bnd, maxiter = 300, popsize = 1000,
                                            args = (res, domen_1, df), disp = True, tol = 0.0000001)
        np.save('modeli/tezine', res_2.x)


    var = np.load('modeli/tezine.npy')
    
    solution = model_new(var, sol, domen_1)
    
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
    plt.ylabel("Udeli")
    plt.show()
    
    plt.savefig('images/slika_{}_{}.png'.format('mean_w',country))
    
    np.arange('2020-03-06', '2020-03', dtype='datetime64[D]')








   



