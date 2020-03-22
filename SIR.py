import scipy.integrate as spint
import numpy as np 
import matplotlib.pyplot as plt 

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt = - beta*S*I
    dI_dt = beta*S*I - gamma*I
    dR_dt = gamma*I

    return ([dS_dt, dI_dt, dR_dt])

if __name__ == "__main__":
    S0 = 0.999
    I0 = 0.001
    R0 = 0

    beta = 0.001
    gamma = 0.2587

    t = np.linspace(0,200,1000)

    solution = spint.odeint(SIR_model, [S0, I0, R0], t, args= (beta, gamma))
    solution = np.array(solution)

    plt.figure(figsize = [10,6])
    # plt.plot(t, solution[:,0], label = "S(t)")
    plt.plot(t, solution[:,1], label = "I(t)")
    plt.plot(t, solution[:,2], label = "R(t)")
    plt.grid()
    plt.legend()
    plt.xlabel("Vreme")
    plt.ylabel("Udeli")
    plt.show()



