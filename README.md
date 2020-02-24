# GPU Parallel Implementation of Dual-Depth Sparse Probabilistic Latent Semantic Analysis for Hyperspectral Unmixing
The Code for "GPU Parallel Implementation of Dual-Depth Sparse Probabilistic Latent Semantic Analysis for Hyperspectral Unmixing". [https://ieeexplore.ieee.org/abstract/document/8809876]
```
J. A. Gallardo, M. E. Paoletti, J. M. Haut, R. Fernandez-Beltran, J. Plaza and A. Plaza.
GPU Parallel Implementation of Dual-Depth Sparse Probabilistic Latent Semantic Analysis for Hyperspectral Unmixing. 
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
DOI: 10.1109/JSTARS.2019.2934011
vol. 12, no. 9, pp. 3156-3167, September 2019.
```

![NVIDIA Profiler](https://github.com/mhaut/gpu_plsa_unmixing_HSI/blob/master/images/Pipeline.jpg)



## Example of use
### Download datasets

```
./retrieveData.sh
```

#### Run code
```
cd cuda
```

#### Probabilistic Latent Semantic Analysis
```
python -u main.py pLSA samson 0 --seed 0
```

#### Dual-Depth Sparse Probabilistic Latent Semantic Analysis
```
python -u main.py  dpLSA samson 0 --dplsa_dim 250 --seed 0
```
