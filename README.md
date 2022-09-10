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

The package provides just two main modules namely,
  `tandem`: 
  `simultaneous`:

<!-- ```pycon
>>> from simuclustfactor import tandem
>>> from tensor import Unfold

>>> X_i_j_k = [[[1,2,3,8],[9,1,2,3],[0,3,6,3]], [[5,1,9,3],[9,1,4,5],[7,2,0,3]],
...            [[7,5,6,8],[3,6,4,7],[3,9,0,7]], [[7,5,6,1],[3,6,5,7],[9,2,8,1]]]
>>> X_i_jk = Unfold(X_i_j_k, mode=1)
>>> I,J,K = 3,4,4  # dimension of the tensor in the fullspace
>>> G,Q,R = 2,3,1  # dimension of tensor in the reduced space
>>> twcfta_res = TWCFTA().fit(X_i_jk=X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
``` -->

```pycon
>>> import numpy as np
>>> from simuclustfactor import tandem
>>> from simuclustfactor import simultaneous
>>> from sklearn.datasets import make_blobs

>>> I,J,K = 40,15,20  # dimensions in the full space.
>>> G,Q,R = 8,4,3  # tensor dimensions in reduced space.  
>>> X_i_jk, y = make_blobs(n_samples=I, centers=G, n_features=J*K, random_state=0)  # generate dataset

>>> twcfta = tandem.TWCFTA(random_state=0,verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

>>> twfcta = tandem.TWFCTA(random_state=0,verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

>>> t3clus = simultaneous.T3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

>>> tfkmeans = simultaneous.TFKMeans(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))

>>> tfkmeans_1 = simultaneous.CT3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=0)

>>> t3clus_1 = simultaneous.CT3Clus(random_state=0, init='random', verbose=True, n_max_iter=10).fit(X_i_jk, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=1)
```