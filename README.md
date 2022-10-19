python-simuclustfactor
===============

Perform simultaneous clustering and factor decomposition in Python for 
three-mode datasets, a library utility.

The main use cases of the library are:

-   performing tandem clustering and factor-decomposition procedures sequentially (TWCFTA).
-   performing tandem factor-decomposition and clustering procedures sequentially (TWFCTA).
-   performing the clustering and factor decomposition procedures simultaneously (T3Clus).
-   performing factor-decomposition and clustering procedures simultaneously (3FKMeans).
-   performing combined T3Clus and 3FKMeans procedures simultaneously (CT3Clus).

Installation
------------

To install the Python library, run:

```shell
pip install simuclustfactor
```

You may consider installing the library only for the current user:

```shell
pip install simuclustfactor --user
```

Library usage
-------------

Implements just two main modules namely,
  - `tandem`: encapsulating TWCFTA and TWFCTA
  - `simultaneous`: encapsulating T3Clus, TFKMeans and CT3Clus

**Data Generation**

```py
import numpy as np
from simuclustfactor import tandem
from simuclustfactor import simultaneous
from simuclustfactor.generate_dataset import GenerateDataset
from simuclustfactor.tensor import Fold, Unfold

I,J,K = 8,5,4  # dimensions in the full space.
G,Q,R = 3,3,2  # tensor dimensions in reduced space.  
data = GenerateDataset(I=I, J=J, K=K, G=G, Q=Q, R=R, centroids_spread=[0,1], noise_mean=0, noise_stdev=0.5, seed=None).additive_noise()

# Extracting the data
Y_g_qr = data.Y_g_qr  # centroids matrix in the reduced space.
Z_i_jk = data.Z_i_jk  # score/centroid matrix in the full-space.
X_i_jk = data.X_i_jk  # dataset with noise.

# Ground-truth parameters
U_i_g = data.U_i_g  # binary stochastic membership matrix
B_j_q = data.B_j_q  # variables factor matrix
C_k_r = data.C_k_r  # occasions factor matrix

# Ground-truth associations
U_i_g = data.U_labels  # array of clusters for objects
B_j_q = data.B_labels  # array of factors for each variable
C_k_r = data.C_labels  # array of factors for each occasion

# Folding generated data matrices into tensors
X_i_j_k = fold(X_i_jk, mode=1, shape=c(K,I,J))
Z_i_j_k = fold(Z_i_jk, mode=1, shape=c(K,I,J))
Y_g_q_r = fold(Y_g_qr, mode=1, shape=c(R,G,Q))
```

### Tandem Models

**TWCFTA**
```py
cf = tandem.TWCFTA(random_state=0,verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

```
**TWFCTA**
```py
twfcta = tandem.TWFCTA(random_state=0,verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

# The following attributes are accessible for the tandem models via the '. operator
twfcta.U_i_g0  # initial membership matrix
twfcta.B_j_q0  # initial variable-component matrix
twfcta.C_k_r0  # initial occasion-component matrix
twfcta.U_i_g  # final membership matrix
twfcta.B_j_q  # final variable-component matrix
twfcta.C_k_r  # final occasion-component matrix

twfcta.Y_g_qr  # The centroids in the reduced space (data matrix).
twfcta.X_i_jk_scaled  # Standardized data matrix.

twfcta.BestTimeElapsed  # Execution time for the best iterate.
twfcta.BestLoop  # Loop that obtained the best iterate.

twfcta.BestKmIteration  # Number of iterations until best iterate for the K-means.
twfcta.BestFaIteration  # Number of iterations until best iterate for the FA.
twfcta.FaConverged  # Flag to check if algorithm converged for the Factor decomposition.
twfcta.KmConverged  # Flag to check if algorithm converged for the K-means.
twfcta.nKmConverges  # Number of loops that converged for the K-means.
twfcta.nFaConverges  # Number of loops that converged for the Factor decomposition.

twfcta.TSS_full  # Total deviance in the full-space.
twfcta.BSS_full  # Between deviance in the reduced-space.
twfcta.RSS_full  # Residual deviance in the reduced-space.
twfcta.TSS_reduced  # Total deviance in the reduced-space.
twfcta.BSS_reduced  # Between deviance in the reduced-space.
twfcta.RSS_reduced  # Residual deviance in the reduced-space.

twfcta.PF_full  # PseudoF computed in the full-space.
twfcta.PF_reduced  # PseudoF computed in the reduced-space.
twfcta.PF  # Actual PseudoF score used to obtain the best loop. PF_reduced for twfcta and PF_full twcfta.

twfcta.Labels  # Object cluster assignments.
twfcta.FsKM  # Objective function values for the KM best iterate.
twfcta.FsFA  # Objective function values for the FA best iterate.
twfcta.Enorm  # Average l2norm of residual norm.
```

**Simultaneous Models**

**TFKMeans**
```py
tfkmeans = simultaneous.TFKMeans(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
```

**TFKMeans**
```py
t3clus = simultaneous.T3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
```

**CT3Clus**
```py
tfkmeans_1 = simultaneous.CT3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=0)

ct3clus = simultaneous.CT3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=0.5)

t3clus_1 = simultaneous.CT3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=1)

# The following attributes are accessible for the simultaneous models via the '.' operator
ct3clus.U_i_g0  # initial membership matrix.
ct3clus.B_j_q0  # initial variable-component matrix.
ct3clus.C_k_r0  # initial occasion-component matrix.
ct3clus.U_i_g  # final membership matrix.
ct3clus.B_j_q  # final variable-component matrix.
ct3clus.C_k_r  # final occasion-component matrix.

ct3clus.Y_g_qr  # Centroids in the reduced space (data matrix).
ct3clus.X_i_jk_scaled  # Standardized data matrix.

ct3clus.BestTimeElapsed  # Execution time for the best iterate.
ct3clus.BestLoop  # Loop that obtained the best iterate.
ct3clus.BestIteration  # Number of iterations until best iterate found.
ct3clus.Converged  # Flag to check if the algorithm converged.
ct3clus.nConverges  # Number of loops that converged.

ct3clus.TSS_full  # Total deviance in the full-space.
ct3clus.BSS_full  # Between deviance in the reduced-space.
ct3clus.RSS_full  # Residual deviance in the reduced-space.
ct3clus.TSS_reduced  # Total deviance in the reduced-space.
ct3clus.BSS_reduced  # Between deviance in the reduced-space.
ct3clus.RSS_reduced  # Residual deviance in the reduced-space.

ct3clus.PF_full  # PseudoF computed in the full-space.
ct3clus.PF_reduced  # PseudoF computed in the reduced-space.
ct3clus.PF  # Weighted PseudoF score used to obtain the best loop.

ct3clus.Labels  # Object cluster assignments.
ct3clus.Fs  # Objective function values for the best iterate.
ct3clus.Enorm  # Average l2norm of residual norm.
```


