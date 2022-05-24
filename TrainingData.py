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

NData = 50000 
Nsize = 6 #5 paramètres du Call + 1 sortie de référence

DataBase = np.zeros((Nsize,NData))


#### On génère aléatoirement des paramètres
S0 = np.random.uniform(0.0, 150.0, (NData))
r = np.random.uniform(0.0, 0.05, (NData))
sg = np.random.uniform(0.0, 0.3, (NData))
T = np.random.uniform(1.0, 10.0, (NData))
K = np.random.uniform(0.0, 150.0, (NData))

#### 
DataBase[0,:] = S0
DataBase[1,:] = r
DataBase[2,:] = sg
DataBase[3,:] = T
DataBase[4,:] = K

for i in range(NData):
    DataBase[5,i] = exactPriceCallBS(S0[i], r[i], sg[i], T[i], K[i])

# Normalisation des données
MeanData = np.mean(DataBase, axis=1)
StdDevData = np.std(DataBase, axis=1)

mu = np.outer(MeanData, np.ones(NData))
std_dev = np.outer(StdDevData, np.ones(NData))

assert(np.shape(DataBase) == np.shape(mu))
assert(np.shape(DataBase) == np.shape(std_dev))

DataBase = (DataBase - mu)/std_dev

print("Moyenne : ", np.mean(DataBase, axis=1))
print("Écart-type : ", np.std(DataBase, axis=1))

# Sauvegarde des données
np.savetxt('MeanData.txt', MeanData)
np.savetxt('StdDevData.txt', StdDevData)
np.savetxt('DataBaseCall.txt', DataBase)