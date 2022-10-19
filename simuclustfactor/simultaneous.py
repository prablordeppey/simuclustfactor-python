### ---- IMPORTING MODULES

import numpy as np
from .tensor import Fold, Unfold
from .utils import EigenVectors, PseudoF_Full, PseudoF_Reduced, _BaseClass, OneKMeans, RandomMembershipMatrix
from .tandem import TWFCTA, TWCFTA
from time import time
from tabulate import tabulate

# === DOCUMENTATIONS
# model initialization configs
_doc_init_args = '''
	Args:
		n_max_iter (integer): number of iterations. Defaults to 10.
		n_loops (integer): number of loops to gurantee global results. Defaults to 10.
		tol (float): tolerance/acceptable error. Defaults to 1e-5.
		seed (boolean): seed for random generations. Defaults to None.
		verbose (boolean): verbosity mode. Defaults to False.
		U_i_g0 (ndarray): (I,G) initial stochastic membership function matrix. Defaults to None.
		B_j_q0 (ndarray): (J,Q) initial component weight matrix for variables. Defaults to None.
		C_k_r0 (ndarray): (K,R) initial component weight matrix for occasions. Defaults to None.
	'''

# inputs to the model
_doc_fit_args = '''
	Args:
		X_i_jk (ndarray): (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		full_tensor_shape (tuple): (I,J,K) dimensions of the original tensor.
		reduced_tensor_shape (tuple): (G,Q,R) dimensions of centroid tensor.
	'''

# accessible result/outputs
_doc_init_attrs = '''
	Attrs:
		- U_i_g0: (I,G) initial stochastic membership function matrix.
		- B_j_q0: (J,Q) initial component weight matrix for variables.
		- C_k_r0: (K,R) initial component weight matrix for occasions.
		- U_i_g: (I,G) final iterate stochastic membership function matrix.
		- B_j_q: (J,Q) final iterate component weight matrix for variables.
		- C_k_r: (K,R) final iterate component weight matrix for occasions.
		- Y_g_qr: (G,QR) matricized version of three-way core tensor.
		- X_i_jk_scaled: is X_i_jk for T3Clus and updated otherwise.
		
		- BestTimeElapsed: time taken for the best iterate.
		- BestLoop: best loop for global result.
		- BestIteration: best iteration for convergence of the procedure.

		- TSSReduced: Total sum of squared deviations for best loop in the reduced space.
		- BSSReduced: Between sum of squared deviations for best loop in the reduced space.
		- RSSReduced: Residual sum of squared deviations for best loop in the reduced space.
		- PFReduced: PsuedoF score from the best loop in the reduced space.
		
		- Labels: cluster labels for the best loop.

		- Fs: A list of objective function values until stopping criteria. 
		- Enorm: Frobenius or L2 norm of residual term from the model.
		- converged: whether the procedure converged or not.
	'''

# === REFERENCES
_doc_refs = '''
	References:
		[1] Vichi, Maurizio & Rocci, Roberto & Kiers, Henk. (2007).
		Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.
		Journal of Classification. 24. 71-98. 10.1007/s00357-007-0006-x. 

		[2] Bro, R. (1998).
		Multi-way analysis in the food industry: models, algorithms, and applications.
		Universiteit van Amsterdam.

		[4] Ledyard Tucker.
		Some mathematical notes on three-mode factor analysis.
		Psychometrika, 31(3):279–311, September 1966.'''

# === DOCUMENTATION FORMATTER FOR METHODS
def _doc_formatter(*sub):
	"""
	elegant docstring formatter
	"""
	def dec(obj):
		obj.__doc__ = obj.__doc__.format(*sub)
		return obj
	return dec

# === The T3CLUS MODEL
# @_doc_formatter(_doc_init_args, _doc_init_attrs)
class T3Clus(_BaseClass):
	"""
	Implements simultaneous version of TWCFTA

	:param integer seed: Seed for random sequence generation. Defaults to None.
	:param bool verbose: Whether to display executions output or not. Defaults to False.
	:param string init: The parameter initialization method. Defaults to 'svd'.
	:param integer n_max_iter: Maximum number of iterations. Defaults to 10.
	:param integer n_loops: Number of initialization to guarantee global results. Defaults to 10.
	:param float tol: Tolerance level/acceptable error. Defaults to 1e-5.
	:param ndarray U_i_g: (IxG) initial stochastic membership function matrix.
	:param ndarray B_j_q: (JxQ) initial component weight matrix for variables.
	:param ndarray C_k_r: (KxR) initial component weight matrix for occasions.

	:return: Initialized T3Clus model.	

	.. note::
		The procedure performs simultaneously the sequential TWCFTA model finding B_j_q and C_k_r 
		such that the between-clusters deviance of the component scores is maximized.
	
	:references:
		[1] Tucker L. (1966)
		Some mathematical notes on three-mode factor analysis
		Psychometrika, 31(3), 279-311. `10.1007/BF02289464 <https://doi.org/10.1007/BF02289464>`_. 
	
		[2] Rocci, Roberto and Vichi, Maurizio (2005)
		Three-Mode Component Analysis with Crisp or Fuzzy Partition of Units
		Journal of Psychometrika. `10.1007/s11336-001-0926-z <https://doi.org/10.1007/s11336-001-0926-z>`_. 

		[3] Vichi, Maurizio & Rocci, Roberto & Kiers, Henk. (2007).
		Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.
		Journal of Classification. 24. 71-98.  `10.1007/s00357-007-0006-x <https://doi.org/10.1007/s00357-007-0006-x>`_.
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
		self.model = CT3Clus(*args, **kwargs)

	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		"""

		:param ndarray X_i_jk: (IxJK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:param tuple full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:param tuple reduced_tensor_shape: (G,Q,R) dimensions of centroid tensor.
		
		:return:

			- U_i_g0 - Initial object membership function matrix
			- B_j_q0 - Initial factor/component matrix for the variables
			- C_k_r0 - Initial factor/component matrix for the occasions
			- U_i_g -  Final/updated object membership function matrix
			- B_j_q - Final/updated factor/component matrix for the variables
			- C_k_r - Final/updated factor/component matrix for the occasions
			- Y_g_qr - Derived centroids in the reduced space (data matrix)
			- X_i_jk_scaled - Standardized dataset matrix
			- BestTimeElapsed - Execution time for the best iterate
			- BestLoop - Loop that obtained the best iterate
			- BestIteration - Iteration yielding the best results
			- Converged - Flag to check if algorithm converged for the K-means
			- nConverges - Number of loops that converged for the K-means
			- TSS_full - Total deviance in the full-space
			- BSS_full - Between deviance in the reduced-space
			- RSS_full - Residual deviance in the reduced-space
			- PF_full - PseudoF in the full-space
			- TSS_reduced - Total deviance in the reduced-space
			- BSS_reduced - Between deviance in the reduced-space
			- RSS_reduced - Residual deviance in the reduced-space
			- PF_reduced - PseudoF in the reduced-space
			- PF - Weighted PseudoF score
			- Labels - Object cluster assignments
			- Fs - Objective function values for the KM best iterate
			- Enorm - Average l2 norm of the residual norm.

		:Example:
			>>> from generate_dataset import GenerateDataset
			>>> seed = 106382
			>>> full_tensor_shape = (8,5,4)
			>>> reduced_tensor_shape = (3,3,2)
			>>> X_i_jk = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise().X_i_jk
			>>> model = T3Clus(verbose=False)
			>>> t3c = model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
			>>> t3c.Y_g_qr
		"""

		return self.model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape, alpha=1)


# === The TFKMeans MODEL
class TFKMeans(_BaseClass):
	"""
	Simultaneous version of TWFCTA - Clustering-Factorial Decomposition (3FKMeans).
	Minimize within cluster deviance of the component scores.
	
	:param integer seed: Seed for random sequence generation. Defaults to None.
	:param bool verbose: Whether to display executions output or not. Defaults to False.
	:param string init: The parameter initialization method. Defaults to 'svd'.
	:param integer n_max_iter: Maximum number of iterations. Defaults to 10.
	:param integer n_loops: Number of initialization to guarantee global results. Defaults to 10.
	:param float tol: Tolerance level/acceptable error. Defaults to 1e-5.
	:param ndarray U_i_g: (IxG) initial stochastic membership function matrix.
	:param ndarray B_j_q: (JxQ) initial component weight matrix for variables.
	:param ndarray C_k_r: (KxR) initial component weight matrix for occasions.

	:return: Initialized 3FKMeans model.

	.. note::
		The procedure performs simultaneously the sequential TWFCTA model finding
		B_j_q and C_k_r such that the within-clusters deviance of the component scores is minimized.
	
	:references:
		[1] Tucker L. (1966)
		Some mathematical notes on three-mode factor analysis
		Psychometrika, 31(3), 279-311. `10.1007/BF02289464 <https://doi.org/10.1007/BF02289464>`_. 

		[2] Vichi, Maurizio and Kiers, Henk A. L. (2001)
		Factorial k-means analysis for two-way data
		Journal of Computational Statistics and Data Analysis.
		`10.1016/S0167-9473(00)00064-5 <https://doi.org/10.1016/S0167-9473(00)00064-5>`_. 

		[3] Vichi, Maurizio & Rocci, Roberto & Kiers, Henk. (2007).
		Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.
		Journal of Classification. 24. 71-98. `10.1007/s00357-007-0006-x <https://doi.org/10.1007/s00357-007-0006-x>`_.
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
		self.model = CT3Clus(*args, **kwargs)


	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		'''
		:param ndarray X_i_jk: (IxJK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:param tuple full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:param tuple reduced_tensor_shape: (G,Q,R) dimensions of centroid tensor.
		
		:return:
			- U_i_g0 - Initial object membership function matrix
			- B_j_q0 - Initial factor/component matrix for the variables
			- C_k_r0 - Initial factor/component matrix for the occasions
			- U_i_g -  Final/updated object membership function matrix
			- B_j_q - Final/updated factor/component matrix for the variables
			- C_k_r - Final/updated factor/component matrix for the occasions
			- Y_g_qr - Derived centroids in the reduced space (data matrix)
			- X_i_jk_scaled - Standardized dataset matrix
			- BestTimeElapsed - Execution time for the best iterate
			- BestLoop - Loop that obtained the best iterate
			- BestIteration - Iteration yielding the best results
			- Converged - Flag to check if algorithm converged for the K-means
			- nConverges - Number of loops that converged for the K-means
			- TSS_full - Total deviance in the full-space
			- BSS_full - Between deviance in the reduced-space
			- RSS_full - Residual deviance in the reduced-space
			- PF_full - PseudoF in the full-space
			- TSS_reduced - Total deviance in the reduced-space
			- BSS_reduced - Between deviance in the reduced-space
			- RSS_reduced - Residual deviance in the reduced-space
			- PF_reduced - PseudoF in the reduced-space
			- PF - Weighted PseudoF score
			- Labels - Object cluster assignments
			- Fs - Objective function values for the KM best iterate
			- Enorm - Average l2 norm of the residual norm.

			:Example:
				>>> from generate_dataset import GenerateDataset
				>>> seed = 106382
				>>> full_tensor_shape = (8,5,4)
				>>> reduced_tensor_shape = (3,3,2)
				>>> X_i_jk = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise().X_i_jk
				>>> model = TFKMeans(verbose=False)
				>>> tfk = model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				>>> tfk.Y_g_qr
		'''
		return self.model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape, alpha=0)


# === The CT3CLUS MODEL
class CT3Clus(_BaseClass):
	"""
	Implements a convex combination of the simultaneous models T3Clus and 3FKMeans integrating
	an alpha value in [0,1] for a weighted result.
	
	:param integer seed: Seed for random sequence generation. Defaults to None.
	:param bool verbose: Whether to display executions output or not. Defaults to False.
	:param string init: The parameter initialization method. Defaults to 'svd'.
	:param integer n_max_iter: Maximum number of iterations. Defaults to 10.
	:param integer n_loops: Number of initialization to guarantee global results. Defaults to 10.
	:param float tol: Tolerance level/acceptable error. Defaults to 1e-5.
	:param ndarray U_i_g: (IxG) initial stochastic membership function matrix.
	:param ndarray B_j_q: (JxQ) initial component weight matrix for variables.
	:param ndarray C_k_r: (KxR) initial component weight matrix for occasions.

	:return: Initialized CT3Clus model.

	:references:
		[1] Tucker L. (1966)
		Some mathematical notes on three-mode factor analysis
		Psychometrika, 31(3), 279-311. `10.1007/BF02289464 <https://doi.org/10.1007/BF02289464>`_. 

		[2] Vichi, Maurizio and Kiers, Henk A. L. (2001)
		Factorial k-means analysis for two-way data
		Journal of Computational Statistics and Data Analysis
		`10.1016/S0167-9473(00)00064-5 <https://doi.org/10.1016/S0167-9473(00)00064-5>`_. 

		[3] Rocci, Roberto and Vichi, Maurizio (2005)
		Three-Mode Component Analysis with Crisp or Fuzzy Partition of Units
		Journal of Psychometrika. `10.1007/s11336-001-0926-z <https://doi.org/10.1007/s11336-001-0926-z>`_. 

		[4] Vichi, Maurizio & Rocci, Roberto & Kiers, Henk. (2007).
		Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.
		Journal of Classification. 24. 71-98.  `10.1007/s00357-007-0006-x <https://doi.org/10.1007/s00357-007-0006-x>`_.
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

	def _check_ct3clus_params(self, full_tensor_shape, reduced_tensor_shape, alpha):
		super()._check_params()

		# alpha check
		if not 0<=alpha<=1:
			raise ValueError(f'alpha must be between [0,1] but got alpha={alpha}')

	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape, alpha=0.5):
		'''
		:param ndarray X_i_jk: (IxJK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:param tuple full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:param tuple reduced_tensor_shape: (G,Q,R) dimensions of centroid tensor.
		:param float alpha: 0<=alpha<=1. alpha=1 for T3Clus, alpha=0 for TFKMeans. alpha=0.5 improve recoverability.
		
		:return:
			- U_i_g0 - Initial object membership function matrix
			- B_j_q0 - Initial factor/component matrix for the variables
			- C_k_r0 - Initial factor/component matrix for the occasions
			- U_i_g -  Final/updated object membership function matrix
			- B_j_q - Final/updated factor/component matrix for the variables
			- C_k_r - Final/updated factor/component matrix for the occasions
			- Y_g_qr - Derived centroids in the reduced space (data matrix)
			- X_i_jk_scaled - Standardized dataset matrix
			- BestTimeElapsed - Execution time for the best iterate
			- BestLoop - Loop that obtained the best iterate
			- BestIteration - Iteration yielding the best results
			- Converged - Flag to check if algorithm converged for the K-means
			- nConverges - Number of loops that converged for the K-means
			- TSS_full - Total deviance in the full-space
			- BSS_full - Between deviance in the reduced-space
			- RSS_full - Residual deviance in the reduced-space
			- PF_full - PseudoF in the full-space
			- TSS_reduced - Total deviance in the reduced-space
			- BSS_reduced - Between deviance in the reduced-space
			- RSS_reduced - Residual deviance in the reduced-space
			- PF_reduced - PseudoF in the reduced-space
			- PF - Weighted PseudoF score
			- Labels - Object cluster assignments
			- Fs - Objective function values for the KM best iterate
			- Enorm - Average l2 norm of the residual norm.
	
		:Example:
			>>> from generate_dataset import GenerateDataset
			>>> seed = 106382
			>>> full_tensor_shape = (8,5,4)
			>>> reduced_tensor_shape = (3,3,2)
			>>> X_i_jk = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise().X_i_jk
			>>> model = CT3Clus(verbose=False)
			>>> ct3 = model.fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
			>>> ct3.Y_g_qr
		'''
		# initializing basic config
		rng = np.random.default_rng(self.seed)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_ct3clus_params(full_tensor_shape, reduced_tensor_shape, alpha)
		# self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_scaled = X_i_jk - X_i_jk.mean(axis=0, keepdims=True)
		X_i_jk = X_scaled/((X_scaled**2).sum(axis=0)/I)**0.5

		I_jk_jk = np.diag(np.ones(J*K))
		I_i_i = np.diag(np.ones(I))

		headers = ['Loop','Iteration','Time Elapsed','BSS Full (%)','BSS Reduced (%)', 'PseudoF Full','PseudoF Reduced', 'PseudoF', 'Convergence']
		if self.verbose: print(tabulate([],headers=headers))

		n_converges = 0  # tracks the number of converges 
		
		for loop in range(1,self.n_loops+1):

			start_time = time()
			iteration = 0
			Fs = []  # objective function values
			converged = False

			# ------------ Start of Initialization ------------
			# given directly as paramters
			U_i_g0 = self.U_i_g
			B_j_q0 = self.B_j_q
			C_k_r0 = self.C_k_r
			
			if self.init == 'random':  # random initialization
				if B_j_q0 is None:
					B_rand = rng.random([J,J])
					B_j_q0 = EigenVectors(B_rand @ B_rand.T, Q)
					del B_rand
				if C_k_r0 is None:
					C_rand = rng.random([K,K])
					C_k_r0 = EigenVectors(C_rand @ C_rand.T, R)
					del C_rand

			# run once if not random
			elif self.init == 'twcfta':
				cft = TWCFTA(seed=self.seed, n_max_iter=self.n_max_iter, n_loops=self.n_loops).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				B_j_q0 = cft.B_j_q
				C_k_r0 = cft.C_k_r
				U_i_g0 = cft.U_i_g
				del cft
			
			elif self.init == 'twfcta':
				fct = TWFCTA(seed=self.seed, n_max_iter=self.n_max_iter, n_loops=self.n_loops).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				B_j_q0 = fct.B_j_q
				C_k_r0 = fct.C_k_r
				U_i_g0 = fct.U_i_g
				del fct
			
			else:  # svd initialization
				
				# permuting X_i_jk by mode
				X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
				X_j_ki = Unfold(X_k_i_j, mode=2)
				X_k_ij = Unfold(X_k_i_j, mode=0)

				# initializing B, C
				if B_j_q0 is None: B_j_q0 = EigenVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = EigenVectors(X_k_ij@X_k_ij.T, R)

			if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# ----------- Start of Objective Function Definition --------------

			U_i_g_init = U_i_g0.copy()
			B_j_q_init = B_j_q0.copy()
			C_k_r_init = C_k_r0.copy()

			# updating X_i_jk
			P = np.kron(C_k_r0 @ C_k_r0.T, B_j_q0 @ B_j_q0.T)
			X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )

			Z_i_qr = U_i_g0 @ np.diag(1/U_i_g0.sum(axis=0)) @ U_i_g0.T @ X_i_jk_N @ np.kron(C_k_r0, B_j_q0)

			F0 = np.linalg.norm(Z_i_qr,2)
			conv = 2*self.tol
			Fs.append(F0)

			# best results
			best_U_i_g = U_i_g0
			best_B_j_q = B_j_q0
			best_C_k_r = C_k_r0
			best_iteration = 1

			# ----------- End of Objective Function Definition --------------

			while (conv > self.tol):
				
				iteration += 1

				# ----------- Start of factor matrices update --------------

				Hu_i_i = U_i_g0 @ np.diag(1/U_i_g0.sum(axis=0)) @ U_i_g0.T

				# permuting X_i_jk_N by mode
				X_k_i_j = Fold(X_i_jk_N, mode=1, shape=(K,I,J))
				X_k_ij = Unfold(X_k_i_j, mode=0)
				X_j_ki = Unfold(X_k_i_j, mode=2)

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(Hu_i_i-alpha*I_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = EigenVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, Hu_i_i-alpha*I_i_i) @ X_k_ij.T
				C_k_r = EigenVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------


				# ----------- Start of objects membership matrix update --------------

				# updating X
				P = np.kron(C_k_r@C_k_r.T, B_j_q@B_j_q.T)
				X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )

				Y_i_qr = X_i_jk_N @ np.kron(C_k_r, B_j_q) # component scores
				U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, seed=self.seed)  # updated membership matrix

				# ----------- End of objects membership matrix update --------------

				# ----------- Start of objective functions update --------------
				Z_i_qr = U_i_g @ np.diag(1/U_i_g.sum(axis=0)) @ U_i_g.T @ Y_i_qr

				F = np.linalg.norm(Z_i_qr,2)
				conv = abs(F-F0)				

				# ----------- End of objective functions update --------------

				# ----------- Start stopping criteria check --------------						

				if F >= F0:
					Fs.append(F)			
					F0 = F
					best_B_j_q = B_j_q
					best_C_k_r = C_k_r
					best_U_i_g = U_i_g
					best_iteration = iteration

				if conv < self.tol:
					converged = True
					n_converges += 1
					break

				if iteration == self.n_max_iter:
					# if self.verbose: print("Maximum iterations reached.")
					break

				# ----------- End stopping criteria check --------------
				
				U_i_g0 = U_i_g
				B_j_q0 = B_j_q
				C_k_r0 = C_k_r

			# ----------- Start of results update for each loop --------------

			time_elapsed = time()-start_time
			
			# updating X
			P = np.kron(best_C_k_r@best_C_k_r.T, best_B_j_q@best_B_j_q.T)
			X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )
			Y_i_qr = X_i_jk_N @ np.kron(best_C_k_r, best_B_j_q)
			Z_i_qr = best_U_i_g @ np.diag(1/best_U_i_g.sum(axis=0)) @ best_U_i_g.T @ Y_i_qr 
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T

			TSS_full = X_i_jk_N.var()*X_i_jk_N.size
			BSS_full = Z_i_jk.var()*Z_i_jk.size
			RSS_full = (X_i_jk_N-Z_i_jk).var()*(X_i_jk_N-Z_i_jk).size

			TSS_reduced = (Y_i_qr @ Y_i_qr.T).trace()  # Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = (Z_i_qr @ Z_i_qr.T).trace()  # Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = ((Y_i_qr-Z_i_qr) @ (Y_i_qr-Z_i_qr).T).trace()  #  (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size			
			
			pseudoF_full = round(PseudoF_Full(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			pseudoF_reduced = round(PseudoF_Reduced(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None

			pseudoF = (1-alpha)*pseudoF_reduced + alpha*pseudoF_full

			# output results
			if self.verbose:
				BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
				BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
				print(tabulate([[]], headers=[loop, best_iteration, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), pseudoF_full, pseudoF_reduced, pseudoF, converged], tablefmt='plain'))

			# tracking the best loop iterates
			if (loop == 1):
				best_PF_simu = pseudoF
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = 1
				converged_simu = converged
				Fs_simu = Fs
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

			if (pseudoF > best_PF_simu):
				best_PF_simu = pseudoF
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = loop
				converged_simu = converged				
				Fs_simu = Fs
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

			# ----------- End of results update for each loop --------------

		# ----------- Start of result update for best loop --------------

		P = np.kron(C_k_r_simu @ C_k_r_simu.T, B_j_q_simu @ B_j_q_simu.T)
		X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )
		Y_i_qr = X_i_jk_N @ np.kron(C_k_r_simu, B_j_q_simu)
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
		self.X_i_jk_scaled = X_i_jk_N

		# total time taken
		self.BestTimeElapsed = best_time_elapsed_simu
		self.BestLoop = loop_simu
		self.BestIteration = iteration_simu
		self.nConverges = n_converges

		# maximum between cluster deviance
		self.TSS_full = TSS_full_simu
		self.BSS_full = BSS_full_simu
		self.RSS_full = RSS_full_simu
		self.PF_full = pseudoF_full_simu
		self.TSS_reduced = TSS_reduced_simu
		self.BSS_reduced = BSS_reduced_simu
		self.RSS_reduced = RSS_reduced_simu
		self.PF_reduced = pseudoF_reduced_simu
		self.PF = best_PF_simu

		# convergence
		self.Enorm = 1/I*np.linalg.norm(X_i_jk_N - Z_i_qr@np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.Fs = Fs_simu  # all objective functional values
		self.converged = converged_simu

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self

