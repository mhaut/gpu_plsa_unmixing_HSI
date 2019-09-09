# Importing PLSA
import time
import numpy as np
from math import ceil

# Cuda imports
import pycuda.autoinit
from pycuda.gpuarray import to_gpu
from kernels import kernels

########################################################################################
##########################     Functions      #########################################
########################################################################################

def initGPU():
    import pycuda.autoinit


def pLSA(lamda, theta, p, X, denominators, shape_image, N, M, K, iters, r1, r2):
    lamda_gpu = to_gpu(lamda)
    theta_gpu = to_gpu(theta)
    p_gpu = to_gpu(p)
    X_gpu = to_gpu(X)
    den_gpu = to_gpu(denominators)
    steps = int(ceil(N/1024.0))
    
    EStep = kernels.get_function("EStep")
    LamdaComputing = kernels.get_function("Lamda")
    ThetaComputing = kernels.get_function("Theta1")
    CalculeDiv = kernels.get_function("Theta2")
    ThetaDivision = kernels.get_function("Theta3")
    
    pycuda.driver.Context.synchronize()
    inicio = time.time()
    
    for i in range(iters):
        EStep(p_gpu, theta_gpu, lamda_gpu, block=(1,1,K), grid=(N,M,1))
        ThetaComputing(theta_gpu, X_gpu, p_gpu, np.float32(r1), np.uint32(steps), np.uint32(N), np.uint32(M), den_gpu, block=(1024,1,1), grid=(1,M,K))
        CalculeDiv(theta_gpu, den_gpu, block=(1, M, 1), grid=(1,1,K))
        ThetaDivision(theta_gpu, den_gpu, block=(1,1,1), grid=(1,M,K))
        LamdaComputing(lamda_gpu, X_gpu, p_gpu,np.uint32(K), np.float32(r2), block=(1,M,1), grid=(N,1,K))
    
    pycuda.driver.Context.synchronize()
    
    endTime = time.time() - inicio
    
    abundances = lamda_gpu.get().reshape(shape_image[0],shape_image[1],K)
    endmembers = theta_gpu.get().T
    
    lamda_gpu.gpudata.free()
    theta_gpu.gpudata.free()
    X_gpu.gpudata.free()
    p_gpu.gpudata.free()
    den_gpu.gpudata.free()
    
    return (abundances,  endmembers, endTime)


        
def dpLSA(lamda, theta, p, X, denominators, shape_image, N, M, K, iters, r1, r2):
    lamda_gpu = to_gpu(lamda)
    theta_gpu = to_gpu(theta)
    p_gpu = to_gpu(p)
    X_gpu = to_gpu(X)
    den_gpu = to_gpu(denominators)
    steps = int(ceil(N/1024.0))
    stepsP = int(ceil(K/50.0))
    
    EStep = kernels.get_function("EStepDPLSA")
    LamdaComputing = kernels.get_function("Lamda")
    ThetaComputing = kernels.get_function("Theta1")
    CalculeDiv = kernels.get_function("Theta2")
    ThetaDivision = kernels.get_function("Theta3")
    
    pycuda.driver.Context.synchronize()
    inicio = time.time()

    for i in range(iters):
        EStep(p_gpu, theta_gpu, lamda_gpu, np.uint32(stepsP), np.uint32(K), block=(1,1,50), grid=(N,M,1))
        ThetaComputing(theta_gpu, X_gpu, p_gpu, np.float32(r1), np.uint32(steps), np.uint32(N), np.uint32(M), den_gpu, block=(1024,1,1), grid=(1,M,K))
        CalculeDiv(theta_gpu, den_gpu, block=(1, M, 1), grid=(1,1,K))
        ThetaDivision(theta_gpu, den_gpu, block=(1,1,1), grid=(1,M,K))
        LamdaComputing(lamda_gpu, X_gpu, p_gpu, np.uint32(K), np.float32(r2), block=(1,M,1), grid=(N,1,K))
    
    pycuda.driver.Context.synchronize()
    
    endTime = time.time() - inicio
    
    abundances = lamda_gpu.get().reshape(shape_image[0],shape_image[1],K)
    endmembers = theta_gpu.get().T
    
    lamda_gpu.gpudata.free()
    theta_gpu.gpudata.free()
    X_gpu.gpudata.free()
    p_gpu.gpudata.free()
    den_gpu.gpudata.free()
    
    return (abundances,  endmembers, endTime)



########################################################################################
########################################################################################
########################################################################################



    
