import sys, time
import numpy as np
import pylab
pylab.seed(0)
from numpy import zeros, int8, log

def EStep():
    for i in range(0, N): # por cada pixel
        for j in range(0, M): # por cada banda
            denominator = 0;
            for k in range(0, K): # por cada endmember
                p[i, j, k] = theta[k, j] * lamda[i, k];
                denominator += p[i, j, k];
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0;
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator;

def MStep():
    # update theta
    for k in range(0, K): # por cada endmember
        denominator = 0
        for j in range(0, M): # de cada banda
            theta[k, j] = 0
            for i in range(0, N): # calcular thetas para cada pixel
                valueaux = X[i, j] * p[i, j, k] - (regularization1 / float(M))
                if valueaux > 0:
                    theta[k, j] += valueaux
            if(k == 0 and j == 0):
                print(theta[0, 0])
            denominator += theta[k, j]
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
    # update lamda
    for i in range(0, N):
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                valueaux = X[i, j] * p[i, j, k] - (regularization2 / float(K))
                if valueaux > 0:
                    lamda[i, k] += valueaux
                denominator += X[i, j];
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator

# calculate the
def LogLikelihood(): # calcular  log likelihood del modelo
    loglikelihood = 0
    for i in range(0, N): # por cada pixel
        for j in range(0, M): # de cada banda
            tmp = 0
            for k in range(0, K): # por cada endmember
                tmp += theta[k, j] * lamda[i, k] # calculo su probabilidad
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    return loglikelihood




########################################################################################
##########################     PARAMETROS      #########################################
########################################################################################
K = 4    # numero de endmembers
maxIteration = 2000
threshold = 10e-6
step2show = 1
step2save = 10
regularization1 = 0 # regularizador1
regularization2 = 0 # regularizador2
path_image = "inputs/pixels_samson.npz"
########################################################################################
########################################################################################
########################################################################################

image  = np.load(path_image)['arr_0']
shape_image = image.shape
N = shape_image[0] * shape_image[1] # num pixeles
M = shape_image[2] # num bands
X = image.reshape(shape_image[0] * shape_image[1], shape_image[2])

# lamda[i, j] : p(zj|di)
lamda = pylab.random([N, K]) # abundances
# theta[i, j] : p(wj|zi)
theta = pylab.random([K, M]) # endmembers
# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K]) # posterior

# normalizacion de parametros lambda y theta
for i in range(0, N):
    normalization = sum(lamda[i, :])
    for j in range(0, K):
        lamda[i, j] /= normalization
for i in range(0, K):
    normalization = sum(theta[i, :])
    for j in range(0, M):
        theta[i, j] /= normalization


oldLoglikelihood = 1; newLoglikelihood = 1
inicio = time.time()
for i in range(1):
    inicio_iteracion = time.time()
    EStep()
    MStep()
    newLoglikelihood = LogLikelihood()
    if i % step2show == 0:
        #print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] ", i+1, " iteration  ", str(newLoglikelihood))
        print("Tiempo/iteracion", time.time() - inicio_iteracion, "Tiempo total", time.time() - inicio, "iteracion", i+1, "likelihood", str(newLoglikelihood))
    if i % step2save == 0:
        np.savez("outputs/abundances_iteration_"+str(i), lamda.reshape(shape_image[0],shape_image[1],K))
        np.savez("outputs/endmembers_iteration_"+str(i), theta.T)
    if ((oldLoglikelihood != 1) and (newLoglikelihood - oldLoglikelihood < threshold)):
        break
    oldLoglikelihood = newLoglikelihood