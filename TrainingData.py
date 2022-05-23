import numpy as np
import openturns as ot

def exactPriceCallBS(S0, r, sg, T, K):
    """
        Fonction calculant le prix d'un call européen dans le monde de B&S en utilisant l'expression analytique.
        -------
        Parameters:
    
    S0 = Spot du sous-jacent.
    r  = Rendement actif sans risque.
    sg = Volatilité du sous-jacent.
    T  = Maturité.
    K  = Strike.

        Returns:
    
    C  = Prix du call européen correspondant.
    """
    alpha = (np.log(S0/K) + (r + sg**2/2)*T)/(sg*np.sqrt(T))
    beta = (np.log(S0/K) + (r - sg**2/2)*T)/(sg*np.sqrt(T))
    N = ot.Normal()

    C = S0 * N.computeCDF(alpha) - np.exp(-r*T) * K * N.computeCDF(beta)  

    return C

NData = 20000 
Nsize = 6 #5 paramètres du Call + 1 sortie de référence

DataBase = np.zeros((Nsize,NData))

for i in range(NData):
    #### On génère aléatoirement des paramètres
    S0 = np.random.uniform(0.0, 150.0)
    r = np.random.uniform(0.0, 0.05)
    sg = np.random.uniform(0.0, 0.3)
    T = np.random.uniform(0.0, 1.0)
    K = np.random.uniform(0.0, 150.0)

    C = exactPriceCallBS(S0, r, sg, T, K)
    #### 
    DataBase[0,i] = S0
    DataBase[1,i] = r
    DataBase[2,i] = sg
    DataBase[3,i] = T
    DataBase[4,i] = K
    DataBase[5,i] = C

np.savetxt('DataBaseCall', DataBase)