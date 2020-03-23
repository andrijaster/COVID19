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
    country = 'China'
    name = "dataset/df_{}.npy".format(country)
    df = np.load(name, allow_pickle = True)
    df = df[:,:]

    I0 = df[0,0]
    R0 = df[0,1]
    M0 = df[0,2]
    S0 = df[0,3]
    
    t = np.linspace(0,70,701)
    time = np.arange(0,df.shape[0])
    index = np.where(np.in1d(t, time))[0]
    bnd = ((0,5),)*4
    domen = np.arange(14,31)
    
    for prekid in domen:
        res = optimize.differential_evolution(Obj_function, bounds = bnd, maxiter = 200, popsize = 1000,
                            args = (t, df, index, prekid, I0, R0, M0, S0), disp = True, tol = 0.0000001)
        np.save("modeli/vektor_{}".format(prekid),res.x)
        beta = res.x[0:2]
        gamma = res.x[2]
        mort = res.x[3]
        
    
        solution = spint.odeint(SIR_model, [I0, R0, M0, S0], t, args= (beta, gamma, mort, prekid))
        solution = np.array(solution)
    
        plt.figure(figsize = [10,6])
        # plt.plot(t, solution[:,3], label = "S(t)")
        plt.plot(t, solution[:,0], label = "I(t)")
        plt.plot(t, solution[:,1], label = "R(t)")
        plt.plot(t, solution[:,2], label = "M(t)")
        
        # plt.plot(time, df[:,3],'o', label = "real S(t)")
        plt.plot(time, df[:,0],'o', label = "real I(t)")
        plt.plot(time, df[:,1],'o',label = "real R(t)")
        plt.plot(time, df[:,2],'o', label = "real M(t)")
        
        plt.grid()
        plt.legend()
        plt.xlabel("Vreme")
        plt.ylabel("Udeli")
        plt.show()
        
        plt.savefig('images/slika_{}.png'.format(prekid))



