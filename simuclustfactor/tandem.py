### ---- IMPORTING MODULES

import numpy as np
from .tensor import Fold, Unfold
from .utils import EigenVectors, PseudoF_Full, PseudoF_Reduced, _BaseClass, OneKMeans, RandomMembershipMatrix
from time import time
from tabulate import tabulate


# === The TWCFTA MODEL
class TWCFTA(_BaseClass):
	"""
	Three-way Clustering-Factorial Tandem model (TWCFTA).
		- Perform KMeans on X_i_jk to obtain membership matrix U_i_g and centroids X_g_jk in full space.
		- Obtain Y_g_qr and C_k_r & B_j_q factor matrices that maximizes the reconstruction of U_i_g.X_g_jk via Tucker2.
	
	:param int n_max_iter: Maximum number of iterations. Defaults to 10.
	:param int n_loops: Number of random initializations to gurantee global results. Defaults to 10.
	:param float tol: Tolerance level/acceptable error. Defaults to 1e-3.
	:param bool seed: Seed for random sequence generation. Defaults to None.
	:param bool verbose: Whether to display executions output or not. Defaults to False.
	:param ndarray U_i_g0: (I,G) initial stochastic membership function matrix.
	:param ndarray B_j_q0: (J,Q) initial component weight matrix for variables.
	:param ndarray C_k_r0: (K,R) initial component weight matrix for occasions.

	:return: Initialized TWCFTA model.

	.. note::
		- This procedure is useful to further interpret the between clusters variability of the data and to understand the variables and/or occasions that most contribute to discriminate the clusters. However, the application of this technique could lead to the masking of variables that are not informative of the clustering structure.

		- Since the Tucker2 model is applied after the clustering, this cannot help select the most relevant information for the clustering in the dataset.

	:references:
		[1] Tucker L. (1966)
		Some mathematical notes on three-mode factor analysis
		Psychometrika, 31(3), 279-311. `10.1007/BF02289464 <https://doi.org/10.1007/BF02289464>`_. 

		[2] Arabie P, Hubert L (1996)
		Advances in Cluster Analysis Relevant to Marketing Research
		In: Gaul, W., Pfeifer, D. (eds) From Data to Knowledge. Studies in Classification, Data Analysis, and Knowledge Organization. Springer, Berlin, Heidelberg.
		`10.1007/978-3-642-79999-0_1 <https://doi.org/10.1007/978-3-642-79999-0_1>`_. 
	"""

	def __init__(
		self,
		*args,
		**kwargs
	):
		super().__init__(
			*args,
			**kwargs
		)

	# Fitting the model & estimating parameters.
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		"""
		:param ndarray X_i_jk: (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:param tuple full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:param tuple reduced_tensor_shape: (G,Q,R) dimensions of centroids tensor.

		:return:
			- U_i_g0 - Initial object membership function matrix.
			- B_j_q0 - Initial factor/component matrix for the variables.
			- C_k_r0 - Initial factor/component matrix for the occasions.
			- U_i_g - Final/updated object membership function matrix.
			- B_j_q - Final/updated factor/component matrix for the variables.
			- C_k_r - Final/updated factor/component matrix for the occasions.
			- Y_g_qr - Derived centroids in the reduced space (data matrix).
			- X_i_jk_scaled - Standardized dataset matrix.
			- BestTimeElapsed - Execution time for the best iterate.
			- BestLoop - Loop that obtained the best iterate.
			- BestKmIteration - Number of iteration until best iterate for the K-means.
			- BestFaIteration - Number of iteration until best iterate for the FA.
			- FaConverged - Flag to check if algorithm converged for the K-means.
			- KmConverged - Flag to check if algorithm converged for the Factor Decomposition.
			- nKmConverges - Number of loops that converged for the K-means.
			- nFaConverges - Number of loops that converged for the Factor decomposition.
			- TSS_full - Total deviance in the full-space.
			- BSS_full - Between deviance in the reduced-space.
			- RSS_full - Residual deviance in the reduced-space.
			- PF_full - PseudoF in the full-space.
			- TSS_reduced - Total deviance in the reduced-space.
			- BSS_reduced - Between deviance in the reduced-space.
			- RSS_reduced - Residual deviance in the reduced-space.
			- PF_reduced - PseudoF in the reduced-space.
			- PF - Actual PseudoF value to obtain best loop.
			- Labels - Object cluster assignments.
			- FsKM - Objective function values for the KM best iterate.
			- FsFA - Objective function values for the FA best iterate.
			- Enorm - Average l2 norm of the residual norm.

		:Example:
			>>> from generate_dataset import GenerateDataset
			>>> seed = 106382
			>>> full_tensor_shape = (8,5,4)
			>>> reduced_tensor_shape = (3,3,2)
			>>> X_i_jk = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise().X_i_jk
			>>> model = TWCFTA(verbose=False)
			>>> cf = model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
			>>> cf.Y_g_qr
		"""

		# ------------ Initialization ------------

		# initializing basic config
		rng = np.random.default_rng(self.seed)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_params()
		# self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_scaled = X_i_jk - X_i_jk.mean(axis=0, keepdims=True)
		X_i_jk = X_scaled/((X_scaled**2).sum(axis=0)/I)**0.5

		headers = ['Loop','Best KM Iter','Best FA Iter','Loop time', 'BSS Full (%)', 'BSS Reduced (%)', 'PseudoF Full', 'PseudoF Reduced', 'KM Converged','FA Converged']
		if self.verbose: print(tabulate([],headers=headers))

		n_faConverges = 0  # number of converges for factor analysis
		n_kmConverges = 0  # number of converges for kmeans

		# ------------ Loop/Run Start ------------

		# Factorial reduction on centroids (via T2 applied to the centroids matrix X_g_jk_bar)
		for loop in range(1, self.n_loops+1):
			
			start_time = time()

			# ------------ KMeans Clustering ------------
			
			# given directly as paramters
			U_i_g0 = self.U_i_g
			km_iter = 0
			km_converged = False
			Fs_km = []

			# random init if not specified
			if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)
			
			U_i_g_init = U_i_g0.copy()

			# initial objective
			X_g_jk0 = np.diag(1/U_i_g0.sum(axis=0)) @ U_i_g0.T @ X_i_jk  # compute centroids matrix
			F0 = np.linalg.norm(U_i_g0@X_g_jk0, 2)  # objective maximization
			Fs_km.append(F0)		

			# clustering on objects (via KMeans applied to X_i_jk)
			conv = 2*self.tol
			
			# iterates init
			best_km_iter = 1
			best_U_i_g = U_i_g0

			while conv > self.tol:
				
				km_iter += 1

				# get random centroids
				U_i_g = OneKMeans(X_i_jk, G, U_i_g=U_i_g0, rng=rng)  # updated membership matrix
				X_g_jk = np.diag(1/U_i_g.sum(axis=0)) @ U_i_g.T @ X_i_jk  # compute centroids matrix

				# check if maximizes orbjective or minimizes the loss
				F = np.linalg.norm(U_i_g @ X_g_jk,2)  # initial objective				
				conv = abs(F-F0)

				if F >= F0:
					F0 = F
					Fs_km.append(F)						
					best_U_i_g = U_i_g
					best_km_iter = km_iter				

				if conv < self.tol:
					km_converged = True
					n_kmConverges += 1
					break

				if km_iter == self.n_max_iter:
					# if self.verbose: print("KM Maximum iterations reached.")
					break

				U_i_g0 = U_i_g
			
			# updated centroids
			X_g_jk = np.diag(1/best_U_i_g.sum(axis=0)) @ best_U_i_g.T @ X_i_jk  # compute centroids matrix						

			# ------------ Factor Decomposition ------------

			fa_converged = False
			fa_iter = 0
			Fs_fa = []  # objective function values

			# matricize centroid tensor
			X_k_g_j = Fold(X_g_jk, mode=1, shape=(K,G,J))
			X_j_kg = Unfold(X_k_g_j, mode=2)
			X_k_gj = Unfold(X_k_g_j, mode=0)

			# as direct input
			B_j_q0 = self.B_j_q
			C_k_r0 = self.C_k_r

			# initialize B and C
			if self.init == 'svd':
				if B_j_q0 is None: B_j_q0 = EigenVectors(X_j_kg@X_j_kg.T, Q)
				if C_k_r0 is None: C_k_r0 = EigenVectors(X_k_gj@X_k_gj.T, R)
			else:  # random initialization	
				if B_j_q0 is None:
					B_rand = rng.random([J,J])
					B_j_q0 = EigenVectors(B_rand @ B_rand.T, Q)
					del B_rand
				if C_k_r0 is None:
					C_rand = rng.random([K,K])
					C_k_r0 = EigenVectors(C_rand @ C_rand.T, R)
					del C_rand
			
			I_g_g = np.eye(G)

			# to return as initializers
			B_j_q_init = B_j_q0.copy()
			C_k_r_init = C_k_r0.copy()

			# updated centroids matrix
			Z_g_jk = X_g_jk @ np.kron(C_k_r0@C_k_r0.T, B_j_q0@B_j_q0.T)

			F0 = np.linalg.norm(Z_g_jk,2)
			Fs_fa.append(F0)
			conv = 2*self.tol
			
			# iterates init
			best_fa_iter = 1
			best_B_j_q = B_j_q0
			best_C_k_r = C_k_r0

			while (conv > self.tol):				

				fa_iter += 1

				# updating B_j_q
				B_j_j = X_j_kg @ np.kron(I_g_g, C_k_r0@C_k_r0.T) @ X_j_kg.T
				B_j_q = EigenVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_gj @ np.kron(B_j_q@B_j_q.T, I_g_g) @ X_k_gj.T
				C_k_r = EigenVectors(C_k_k, R)

				# updated centroids matrix
				Z_g_jk = X_g_jk @ np.kron(C_k_r@C_k_r.T, B_j_q@B_j_q.T)

				# compute L2 norm of reconstruction error
				F = np.linalg.norm(Z_g_jk,2)				
				conv = abs(F-F0)

				# convergence check
				if F >= F0:
					Fs_fa.append(F)
					F0 = F
					best_B_j_q = B_j_q
					best_C_k_r = C_k_r
					best_fa_iter = fa_iter

				if (conv < self.tol):
					fa_converged = True
					n_faConverges += 1
					break

				if (fa_iter == self.n_max_iter):
					# if self.verbose: print("FA Maximum iterations reached.")
					break

				B_j_q0 = B_j_q
				C_k_r0 = C_k_r

			# ----------- Compute metrics for loop/run --------------

			time_elapsed = time()-start_time

			# updating X
			Y_i_qr = X_i_jk @ np.kron(best_C_k_r, best_B_j_q)
			Z_i_qr = best_U_i_g @ np.diag(1/best_U_i_g.sum(axis=0)) @ best_U_i_g.T @ Y_i_qr
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T

			TSS_full = (X_i_jk @ X_i_jk.T).trace() # X_i_jk@X_i_jk.size
			BSS_full = (Z_i_jk @ Z_i_jk.T).trace() # Z_i_jk.var()*Z_i_jk.size
			RSS_full = ((X_i_jk-Z_i_jk)@(X_i_jk-Z_i_jk).T).trace() # (X_i_jk-Z_i_jk).var()*(X_i_jk-Z_i_jk).size

			TSS_reduced = (Y_i_qr @ Y_i_qr.T).trace()  # Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = (Z_i_qr @ Z_i_qr.T).trace()  # Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = ((Y_i_qr-Z_i_qr) @ (Y_i_qr-Z_i_qr).T).trace()  #  (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size

			# pseudoF scores
			pseudoF_full = PseudoF_Full(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape) if G not in [1,I] else None
			pseudoF_reduced = PseudoF_Reduced(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape) if G not in [1,I] else None

			# output results
			if self.verbose:
				BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
				BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
				print(tabulate([[]], headers=[loop, best_km_iter, best_fa_iter, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), round(pseudoF_full,4), round(pseudoF_reduced,4), km_converged, fa_converged], tablefmt='plain'))

			# tracking the best loop iterates
			if (loop == 1):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				km_iter_simu = best_km_iter
				fa_iter_simu = best_fa_iter
				loop_simu = 1
				km_converged_simu = km_converged
				fa_converged_simu = fa_converged
				Fs_fa = Fs_fa
				Fs_km = Fs_km
				pseudoF_full_simu = pseudoF_full
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				pseudoF_reduced_simu = pseudoF_reduced
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			if (pseudoF_full > pseudoF_full_simu):
				pseudoF_full_simu = pseudoF_full
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				km_iter_simu = best_km_iter  # number of iterations until convergence
				fa_iter_simu = best_fa_iter
				loop_simu = loop  # best loop so far
				km_converged_simu = km_converged  # if there was a convergence
				fa_converged_simu = fa_converged  # if there was a convergence
				Fs_fa = Fs_fa  # objective function values for FA
				Fs_km = Fs_km
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				pseudoF_reduced_simu = pseudoF_reduced
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

		# ----------- Result update for best loop --------------

		Y_i_qr = X_i_jk @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.diag(1/U_i_g_simu.sum(axis=0)) @ U_i_g_simu.T @ Y_i_qr
		Z_i_qr = U_i_g_simu @ Y_g_qr

		# factor matrices and centroid matrices		
		self.U_i_g0 = U_i_g_init_simu
		self.B_j_q0 = B_j_q_init_simu
		self.C_k_r0 = C_k_r_init_simu
		self.U_i_g = U_i_g_simu
		self.B_j_q = B_j_q_simu
		self.C_k_r = C_k_r_simu
		self.Y_g_qr = Y_g_qr
		self.X_i_jk_scaled = X_i_jk

		# total time taken
		self.BestTimeElapsed = best_time_elapsed_simu
		self.BestLoop = loop_simu
		self.BestKmIteration = km_iter_simu
		self.BestFaIteration = fa_iter_simu
		self.FaConverged = fa_converged_simu
		self.KmConverged = km_converged_simu
		self.nKmConverges = n_kmConverges
		self.nFaConverges = n_faConverges

		# maximum between cluster deviance
		self.TSS_full = TSS_full_simu
		self.BSS_full = BSS_full_simu
		self.RSS_full = RSS_full_simu
		self.PF_full = pseudoF_full
		self.TSS_reduced = TSS_reduced_simu
		self.BSS_reduced = BSS_reduced_simu
		self.RSS_reduced = RSS_reduced_simu
		self.PF_reduced = pseudoF_reduced
		self.PF = pseudoF_full

		# Error in model
		self.Enorm = 1/I*np.linalg.norm(X_i_jk - Z_i_qr @ np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.FsKM = Fs_km  # objective values for kmeans
		self.FsFA = Fs_fa  # objective values for factor decomposition

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		return self


# === The TWFCTA MODEL
# @_doc_formatter(_doc_init_args)
class TWFCTA(_BaseClass):
	"""
	The procedure implements sequential factorial decomposition and clustering.
		
		- The technique performs Tucker2 decomposition on the X_i_jk matrix to obtain the matrix of component scores Y_i_qr with component weights matrices B_j_q and C_k_r.
		- The K-means clustering algorithm is then applied to the component scores matrix Y_i_qr to obtain the desired core centroids matrix Y_g_qr and its associated stochastic membership function matrix U_i_g.

	:param int n_max_iter: Maximum number of iterations. Defaults to 10.
	:param int n_loops: Number of random initializations to gurantee global results. Defaults to 10.
	:param float tol: Tolerance level/acceptable error. Defaults to 1e-3.
	:param bool seed: Seed for random sequence generation. Defaults to None.
	:param bool verbose: Whether to display executions output or not. Defaults to False.
	:param ndarray U_i_g0: (I,G) initial stochastic membership function matrix.
	:param ndarray B_j_q0: (J,Q) initial component weight matrix for variables.
	:param ndarray C_k_r0: (K,R) initial component weight matrix for occasions.

	:return: Initialized TWFCTA model.

	.. note:: 
		
		- The technique helps interpret the within clusters variability of the data. The Tucker2 tends to explain most of the total variation in the dataset. Hence, the variance of variables that do not contribute to the clustering structure in the dataset is also included.
		- The Tucker2 dimensions may still mask some essential clustering structures in the dataset.

	:references:
		[1] Tucker L. (1966)
		Some mathematical notes on three-mode factor analysis
		Psychometrika, 31(3), 279-311. `10.1007/BF02289464 <https://doi.org/10.1007/BF02289464>`_. 

		[2] Arabie P, Hubert L (1996)
		Advances in Cluster Analysis Relevant to Marketing Research
		In: Gaul, W., Pfeifer, D. (eds) From Data to Knowledge. Studies in Classification, Data Analysis, and Knowledge Organization. Springer, Berlin, Heidelberg.
		`10.1007/978-3-642-79999-0_1 <https://doi.org/10.1007/978-3-642-79999-0_1>`_.
	"""

	def __init__(
		self,
		*args,
		**kwargs
	):
		super().__init__(
			*args,
			**kwargs
		)

	# @_doc_formatter(_doc_init_attrs, _doc_refs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		"""
		:param ndarray X_i_jk: (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:param tuple full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:param tuple reduced_tensor_shape: (G,Q,R) dimensions of centroids tensor.
		
		:return:
			- U_i_g0 - Initial object membership function matrix.
			- B_j_q0 - Initial factor/component matrix for the variables.
			- C_k_r0 - Initial factor/component matrix for the occasions.
			- U_i_g - Final/updated object membership function matrix.
			- B_j_q - Final/updated factor/component matrix for the variables.
			- C_k_r - Final/updated factor/component matrix for the occasions.
			- Y_g_qr - Derived centroids in the reduced space (data matrix).
			- X_i_jk_scaled - Standardized dataset matrix.
			- BestTimeElapsed - Execution time for the best iterate.
			- BestLoop - Loop that obtained the best iterate.
			- BestKmIteration - Number of iteration until best iterate for the K-means.
			- BestFaIteration - Number of iteration until best iterate for the FA.
			- FaConverged - Flag to check if algorithm converged for the K-means.
			- KmConverged - Flag to check if algorithm converged for the Factor Decomposition.
			- nKmConverges - Number of loops that converged for the K-means.
			- nFaConverges - Number of loops that converged for the Factor decomposition.
			- TSS_full - Total deviance in the full-space.
			- BSS_full - Between deviance in the reduced-space.
			- RSS_full - Residual deviance in the reduced-space.
			- PF_full - PseudoF in the full-space.
			- TSS_reduced - Total deviance in the reduced-space.
			- BSS_reduced - Between deviance in the reduced-space.
			- RSS_reduced - Residual deviance in the reduced-space.
			- PF_reduced - PseudoF in the reduced-space.
			- PF - Actual PseudoF value to obtain best loop.
			- Labels - Object cluster assignments.
			- FsKM - Objective function values for the KM best iterate.
			- FsFA - Objective function values for the FA best iterate.
			- Enorm - Average l2 norm of the residual norm.

		:Example:
			>>> from generate_dataset import GenerateDataset
			>>> seed = 106382
			>>> full_tensor_shape = (8,5,4)
			>>> reduced_tensor_shape = (3,3,2)
			>>> X_i_jk = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise().X_i_jk
			>>> model = TWFCTA(verbose=False)
			>>> fc = model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
			>>> fc.Y_g_qr
		"""

		# ------------ Initialization ------------

		# initializing basic config
		rng = np.random.default_rng(self.seed)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_params()
		# self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)

		# declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_scaled = X_i_jk - X_i_jk.mean(axis=0, keepdims=True)
		X_i_jk = X_scaled/((X_scaled**2).sum(axis=0)/I)**0.5

		# matricize centroid tensor
		X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
		X_j_ki = Unfold(X_k_i_j, mode=2)
		X_k_ij = Unfold(X_k_i_j, mode=0)

		I_i_i = np.diag(np.ones(I))  # identity matrix

		headers = ['Loop', 'FA Iter', 'KM Iter', 'Loop time', 'BSS Full (%)', 'BSS Reduced (%)', 'PseudoF Full', 'PseudoF Reduced', 'FA Converged', 'KM Converged']
		if self.verbose == True: print(tabulate([],headers=headers))

		n_faConverges = 0  # number of converges for factor analysis
		n_kmConverges = 0  # number of converges for kmeans

		# number of loops
		for loop in range(1, self.n_loops+1):
			
			start_time = time()

			# ------------ Start of initialization for FA ------------

			fa_iter = 0
			Fs_fa = []  # objective function values
			converged = False

			# as direct input
			B_j_q0 = self.B_j_q
			C_k_r0 = self.C_k_r

			# initialize B and C
			if self.init == 'svd':
				if B_j_q0 is None: B_j_q0 = EigenVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = EigenVectors(X_k_ij@X_k_ij.T, R)
			else:  # random initialization
				if B_j_q0 is None:
					B_rand = rng.random([J,J])
					B_j_q0 = EigenVectors(B_rand @ B_rand.T, Q)
					del B_rand
				if C_k_r0 is None:
					C_rand = rng.random([K,K])
					C_k_r0 = EigenVectors(C_rand @ C_rand.T, R)
					del C_rand

			# to return as initializers
			B_j_q_init = B_j_q0
			C_k_r_init = C_k_r0

			# updated centroids matrix
			Z_i_jk = X_i_jk @ np.kron(C_k_r0@C_k_r0.T, B_j_q0@B_j_q0.T)

			F0 = np.linalg.norm(Z_i_jk,2)
			conv = 2*self.tol
			Fs_fa.append(F0)
			fa_converged = False

			# iterates init
			best_fa_iter = 1
			best_B_j_q = B_j_q0
			best_C_k_r = C_k_r0
			
			while (conv > self.tol):

				fa_iter += 1

				# ----------- Start of factor matrices update --------------

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(I_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = EigenVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, I_i_i) @ X_k_ij.T
				C_k_r = EigenVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------

				# ----------- Start of objective functions update --------------

				# updated centroids matrix
				Z_i_jk = X_i_jk @ np.kron(C_k_r@C_k_r.T, B_j_q@B_j_q.T)
 
				# compute L2 norm of reconstruction error
				F = np.linalg.norm(Z_i_jk, 2)
				conv = abs(F-F0)

				# ----------- End of objective functions update --------------

				# ----------- Start stopping criteria check --------------

				if F >= F0:
					Fs_fa.append(F)
					F0 = F
					best_B_j_q = B_j_q
					best_C_k_r = C_k_r
					best_fa_iter = fa_iter

				if (conv < self.tol):
					fa_converged = True
					n_faConverges += 1
					break

				if (fa_iter == self.n_max_iter):
					# if self.verbose: print("FA Maximum iterations reached.")
					break

				B_j_q0 = B_j_q
				C_k_r0 = C_k_r

			# ------------ Start of K-means clustering ------------

			Y_i_qr = X_i_jk @ np.kron(best_C_k_r, best_B_j_q)
			km_iter = 0
			km_converged = False
			Fs_km = []

			# given directly as paramters
			U_i_g0 = self.U_i_g
			if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)
			
			U_i_g_init = U_i_g0.copy()  # to return as U_i_g0

			# initial objective
			Y_g_qr0 = np.diag(1/U_i_g0.sum(axis=0)) @ U_i_g0.T @ Y_i_qr  # compute centroids matrix
			F0 = np.linalg.norm(U_i_g0@Y_g_qr0,2)  # residual matrix
			Fs_km.append(F0)

			# clustering on objects (via KMeans applied to X_i_jk)
			conv = 2*self.tol

			# iterates init
			best_km_iter = 1
			best_U_i_g = U_i_g0
			
			# print('initial',U_i_g0)
			while conv > self.tol:
				
				km_iter += 1

				# centroids update
				U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, rng=rng)  # updated membership matrix
				Z_i_qr = U_i_g @ np.diag(1/U_i_g.sum(axis=0)) @ U_i_g.T @ Y_i_qr  # compute centroids matrix
				
				# check if maximizes objective
				F = np.linalg.norm(Z_i_qr, 2)  # residual matrix
				conv = abs(F-F0)
				
				if F >= F0:
					Fs_km.append(F)
					F0 = F			
					best_U_i_g = U_i_g
					best_km_iter = km_iter

				if conv < self.tol:
					km_converged = True
					n_kmConverges += 1
					break

				if km_iter == self.n_max_iter:
					# if self.verbose: print("KM Maximum iterations reached.")
					break

				U_i_g0 = U_i_g  # perform computation with updated U

			
			# ----------- Compute metrics for loop/run --------------

			time_elapsed = time()-start_time

			# updating X
			X_i_jk_N = X_i_jk @ np.kron(best_C_k_r@best_C_k_r.T, best_B_j_q@best_B_j_q.T)
			Y_i_qr = X_i_jk_N @ np.kron(best_C_k_r, best_B_j_q)
			Z_i_qr = best_U_i_g @ np.diag(1/best_U_i_g.sum(axis=0)) @ best_U_i_g.T @ Y_i_qr
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T		
			
			TSS_full = (X_i_jk_N @ X_i_jk_N.T).trace() # X_i_jk@X_i_jk.size
			BSS_full = (Z_i_jk @ Z_i_jk.T).trace() # Z_i_jk.var()*Z_i_jk.size
			RSS_full = ((X_i_jk_N-Z_i_jk)@(X_i_jk_N-Z_i_jk).T).trace() # (X_i_jk_N-Z_i_jk).var()*(X_i_jk_N-Z_i_jk).size

			TSS_reduced = (Y_i_qr @ Y_i_qr.T).trace()  # Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = (Z_i_qr @ Z_i_qr.T).trace()  # Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = ((Y_i_qr-Z_i_qr) @ (Y_i_qr-Z_i_qr).T).trace()  #  (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size

			pseudoF_full = PseudoF_Full(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape) if G not in [1,I] else None
			pseudoF_reduced = PseudoF_Reduced(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape) if G not in [1,I] else None

			# pseudoF and output results
			if self.verbose:
				BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
				BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
				print(tabulate([[]], headers=[loop, best_fa_iter, best_km_iter, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), round(pseudoF_full,4), round(pseudoF_reduced,4), fa_converged, km_converged], tablefmt='plain'))

			# tracking the best loop iterates
			if (loop == 1):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				km_iter_simu = best_km_iter
				fa_iter_simu = best_fa_iter
				loop_simu = 1
				km_converged_simu = km_converged
				fa_converged_simu = fa_converged
				Fs_fa = Fs_fa
				Fs_km = Fs_km
				pseudoF_full_simu = pseudoF_full
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				pseudoF_reduced_simu = pseudoF_reduced
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed


			if (pseudoF_reduced > pseudoF_reduced_simu):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				km_iter_simu = best_km_iter
				fa_iter_simu = best_fa_iter
				loop_simu = loop
				km_converged_simu = km_converged
				fa_converged_simu = fa_converged
				Fs_fa = Fs_fa
				Fs_km = Fs_km
				pseudoF_full_simu = pseudoF_full
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				pseudoF_reduced_simu = pseudoF_reduced
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

		# ----------- Result update for best loop --------------
		X_i_jk_N = X_i_jk @ np.kron(best_C_k_r@best_C_k_r.T, best_B_j_q@best_B_j_q.T)
		Y_i_qr = X_i_jk_N @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.diag(1/U_i_g_simu.sum(axis=0)) @ U_i_g_simu.T @ Y_i_qr
		Z_i_qr = U_i_g_simu @ Y_g_qr

		# factor matrices and centroid matrices
		self.U_i_g = U_i_g_simu
		self.U_i_g0 = U_i_g_init_simu
		self.B_j_q0 = B_j_q_init_simu
		self.C_k_r0 = C_k_r_init_simu
		self.B_j_q = B_j_q_simu
		self.C_k_r = C_k_r_simu
		self.Y_g_qr = Y_g_qr
		self.X_i_jk_scaled = X_i_jk_N

		# total time taken
		self.BestTimeElapsed = best_time_elapsed_simu
		self.BestLoop = loop_simu
		self.BestKMIteration = km_iter_simu
		self.BestFAIteration = fa_iter_simu
		self.FaConverged = fa_converged_simu
		self.KmConverged = km_converged_simu
		self.nKmConverges = n_kmConverges
		self.nFaConverges = n_faConverges

		# maximum between cluster deviance
		self.TSS_full = TSS_full_simu
		self.BSS_full = BSS_full_simu
		self.RSS_full = RSS_full_simu
		self.PF_full = pseudoF_full
		self.TSS_reduced = TSS_reduced_simu
		self.BSS_reduced = BSS_reduced_simu
		self.RSS_reduced = RSS_reduced_simu
		self.PF_reduced = pseudoF_reduced
		self.PF = pseudoF_reduced

		# Error in model
		self.FsKM = Fs_km  # all objective values
		self.FsFA = Fs_fa  # all objectiveh values for Factor Analysis
		self.Enorm = 1/I*np.linalg.norm(X_i_jk - Z_i_qr @ np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		
		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self
