import scipy.integrate as spint
import numpy as np 
import matplotlib.pyplot as plt 


def obj_function(var, t, beta, gamma, mort, S0, I0, R0, M0):
    beta = var[0]
    gamma = var[1]
    mort = var[2]
    
    solution = spint.odeint(SIR_model, [S0, I0, R0, M0], t, args= (beta, gamma, mort))


def SIR_model(y, t, beta, gamma, mort):
    S, I, R, M = y
    dS_dt = - beta*S*I
    dI_dt = beta*S*I - gamma*I - mort*I
    dR_dt = gamma*I
    dM_dt = mort*I

    return ([dS_dt, dI_dt, dR_dt, dM_dt])

if __name__ == "__main__":
    S0 = 0.999
    I0 = 0.001
    R0 = 0
    M0 = 0

    beta = 0.01
    gamma = 0.2587
    mort = 0.029925

    t = np.linspace(0,200,1000)

    solution = spint.odeint(SIR_model, [S0, I0, R0, M0], t, args= (beta, gamma, mort))
    solution = np.array(solution)

    plt.figure(figsize = [10,6])
    # plt.plot(t, solution[:,0], label = "S(t)")
    plt.plot(t, solution[:,1], label = "I(t)")
    plt.plot(t, solution[:,2], label = "R(t)")
    plt.plot(t, solution[:,3], label = "M(t)")
    plt.grid()
    plt.legend()
    plt.xlabel("Vreme")
    plt.ylabel("Udeli")
    plt.show()



