import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import numpy.random as rand

from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import softmax
from numpy import zeros
import scipy.spatial.distance as sp_dist

def set_seed(seed):
    rand.seed(seed)


def load_dataset(dataset):
    if dataset == 'samson':
        path_image = "inputs/samson_1.mat"; key = 'V'
        path_GT = "groundtruths/samson_gt.mat"
        K_img = 3
    elif dataset == 'jasper':
        path_image = "inputs/jasperRidge2_R198.mat"; key = 'Y'
        path_GT = "groundtruths/jasper_gt.mat"
        K_img = 4
    elif dataset == 'urban':
        path_image = "inputs/Urban_R162.mat"; key = 'Y'
        path_GT = "groundtruths/urban_gt.mat"
        K_img = 4
    elif dataset == 'cuprite':
        path_image = "inputs/Cuprite_S1_R188.mat"; key = 'Y'
        path_GT = "groundtruths/cuprite_gt.mat"
        K_img = 12

    image = scipy.io.loadmat(path_image)[key].T
    nRow  = scipy.io.loadmat(path_image)['nRow'][0][0]
    nCol  = scipy.io.loadmat(path_image)['nCol'][0][0]
    image = np.transpose(image.reshape(nCol, nRow, image.shape[1]), (1,0,2))
    return image, path_GT, K_img

def generate_inputs(image, shape_image, K):    
    N = shape_image[0] * shape_image[1] # num pixeles
    M = shape_image[2] # num bands
    X = image.reshape(shape_image[0] * shape_image[1], shape_image[2])
    

    # lamda[i, j] : p(zj|di)
    lamda = rand.uniform(size=(N, K)) # abundances
    # theta[i, j] : p(wj|zi)
    theta = rand.random([K, M]) # endmembers
    # p[i, j, k] : p(zk|di,wj)
    p = zeros([N, M, K]) # posterior
    denominators = zeros(K)
    
    # normalizacion de parametros lambda y theta
    for i in range(0, N):
        normalization = sum(lamda[i, :])
        for j in range(0, K):
            lamda[i, j] /= normalization
    for i in range(0, K):
        normalization = sum(theta[i, :])
        for j in range(0, M):
            theta[i, j] /= normalization
    lamda = lamda.astype(np.float32)
    theta = theta.astype(np.float32)
    p = p.astype(np.float32)
    X = X.astype(np.float32)
    
    return p, X, denominators, theta, lamda, N, M

def get_metrics(endmembersPredicted,  image, exec_id, abundancesPredicted=None, path_abundances_GT=None, show_images=False):
    if show_images == False:
        matplotlib.use('Agg')

    K = endmembersPredicted.shape[1]
    endmembersGT = scipy.io.loadmat(path_abundances_GT)['M']
    if image == 'cuprite':
        bands = scipy.io.loadmat(path_abundances_GT)['slctBnds'][0,:]
        endmembersGT = endmembersGT[bands]
        softmaxed = softmax(endmembersGT.T)
        endmembersGT = softmaxed.T
    
    rmse = 0.0
    sad = 0.0
    
    if (path_abundances_GT != None):
        
        # Pair predicted/true equal endmembers before SAD
        endm_s1 = endmembersPredicted
        endm_gt = endmembersGT

        dists = []
        for col in range(endm_s1.shape[1]):
            act_sim = []
            row  = endm_s1[:,col]
            for col2 in range(endm_gt.shape[1]):
                row2 = endm_gt[:, col2]
                act_sim.append(sp_dist.cosine(row, row2))
            dists.append(act_sim)
        dists = np.array(dists)
        new_classes = [0] * K
        en2 = copy.deepcopy(endmembersPredicted)
        for i in range(K):
            (fil,col) = np.unravel_index(dists.argmin(), dists.shape)
            endmembersPredicted[:,col] = en2[:,fil]
            new_classes[fil] = col
            dists[:,col] = 100000
            dists[fil,:] = 100000
        del en2, new_classes, dists, endm_gt, endm_s1

        from numpy.linalg import norm
    
        cos_sim = 0
        for i in range(K):
            b = endmembersGT[:,i]
            a = endmembersPredicted[:,i]
            cos_sim += np.arccos(np.dot(a,b)/(norm(a)*norm(b)))
        sad = cos_sim / float(K)
        if ('A' in scipy.io.loadmat(path_abundances_GT).keys()):
            abundancesGT = scipy.io.loadmat(path_abundances_GT)['A'].T
            abundancesGT = \
                np.transpose(\
                    abundancesGT.reshape(abundancesPredicted.shape[1], abundancesPredicted.shape[0], abundancesGT.shape[1]), (1,0,2))
    
            # Pair predicted/true equal abundances before RMSE
            image_s1 = abundancesPredicted.reshape(-1, K)
            image_gt = abundancesGT.reshape(-1, K)
    
            dists = []
            for col in range(image_s1.shape[1]):
                act_sim = []
                row  = image_s1[:,col]
                for col2 in range(image_gt.shape[1]):
                    row2 = image_gt[:,col2]
                    act_sim.append(sp_dist.cosine(row, row2))
                dists.append(act_sim)
            dists = np.array(dists)
            new_classes = [0] * K
            ab2 = copy.deepcopy(abundancesPredicted)
            for i in range(K):
                (fil,col) = np.unravel_index(dists.argmin(), dists.shape)
                abundancesPredicted[:,:,col] = ab2[:,:,fil]
                new_classes[fil] = col
                dists[:,col] = 100000
                dists[fil,:] = 100000
            del ab2, new_classes, dists, image_gt, image_s1
    
            rmse = np.sqrt(mean_squared_error(abundancesGT.reshape(-1,K),
                                            abundancesPredicted.reshape(-1,K)))
    
            mosaicPred = abundancesPredicted[:,:,0]; mosaicGT = abundancesGT[:,:,0]
            for i in range(1, K):
                mosaicPred = np.hstack((mosaicPred, abundancesPredicted[:,:,i]))
                mosaicGT   = np.hstack((mosaicGT, abundancesGT[:,:,i]))
            mosaicFinal = np.vstack((mosaicPred, mosaicGT))
            if show_images:
                plt.imshow(mosaicFinal)
                plt.show()
                plt.clf()
            else:
                plt.imsave(('outputs/images/abundances_' + image + '_' + exec_id + '.png'), mosaicFinal)
        else:
            mosaicPred = abundancesPredicted[:,:,0]
            for i in range(1, K):
                mosaicPred = np.hstack((mosaicPred, abundancesPredicted[:,:,i]))
            if show_images:
                plt.imshow(mosaicPred)
                plt.show()
                plt.clf()
            else:
                plt.imsave(('outputs/images/abundances_' + image + '_' + exec_id + '.png'), mosaicPred)
        
    if image == 'cuprite':
        endmembersPredicted = endmembersPredicted[3:, :]

    plt.plot(endmembersPredicted)

    if show_images:
        plt.show()
    else:
        plt.savefig(('outputs/images/endmembers_' + image + '_' + exec_id + '.png'), bbox_inches='tight', pad_inches=0.2, dpi=200)
        
    return rmse, sad# -*- coding: utf-8 -*-

