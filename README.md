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

The module provides just five classes, `tabulate`, which takes a list of
lists or another tabular data type as the first argument, and outputs a
nicely formatted plain-text table:

```pycon
>>> from simuclustfactor import tandem

>>> X_i_j_k = [[[1,2,3,8],[9,1,2,3],[0,3,6,3]], [[5,1,9],[9,1,4]],
...          [[7,5,6],[3,6,7]], [[7,5,6],[3,6,7]]]
>>> I,J,K = 3,4,4
>>> G,Q,R = 2,3,1
>>> twcfta_res = TWCFTA().fit(X=X_i_jk)
```
