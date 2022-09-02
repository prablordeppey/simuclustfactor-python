### ---- IMPORTING MODULES

import numpy as np
from .tensor import Fold, Unfold
from .utils import SingularVectors, PseudoF, _BaseClass, OneKMeans, RandomMembershipMatrix
from time import time
from tabulate import tabulate


# === DOCUMENTATIONS
# model initialization configs
_doc_init_args = '''
	Initialization Args:
		:n_max_iter: Maximum number of iterations. Defaults to 10.
		:n_loops: Number of random initializations to gurantee global results. Defaults to 10.
		:tol: Tolerance level/acceptable error. Defaults to 1e-5.
		:random_state: Seed for random sequence generation. Defaults to None.
		:verbose: Whether to display executions output or not. Defaults to False.
		:U_i_g0: (I,G) initial stochastic membership function matrix.
		:B_j_q0: (J,Q) initial component weight matrix for variables.
		:C_k_r0: (K,R) initial component weight matrix for occasions.'''

# inputs to the model
_doc_fit_args = '''
	Model Fitting Args:
		:X_i_jk: (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
		:full_tensor_shape: (I,J,K) dimensions of the original tensor.
		:reduced_tensor_shape: (G,Q,R) dimensions of centroids tensor.'''

# accessible result/outputs
_doc_init_attrs = '''
	Attributes:
		:U_i_g0: (I,G) initial stochastic membership function matrix.
		:B_j_q0: (J,Q) initial component weight matrix for variables.
		:C_k_r0: (K,R) initial component weight matrix for occasions.
		:U_i_g: (I,G) final iterate stochastic membership function matrix.
		:B_j_q: (J,Q) final iterate component weight matrix for variables.
		:C_k_r: (K,R) final iterate component weight matrix for occasions.
		:Y_g_qr: (G,QR) matricized version of three-way core tensor.
		:X_i_jk_scaled: Scaled X_i_jk
	
		:BestTimeElapsed: Time taken for the best iterate.
		:BestLoop: Best loop for global result.
		:BestKMIteration: Best iteration for convergence of the K-means procedure.
		:BestFAIteration: Best iteration for convergence of the factor decomposition procedure.
		:FaConverged: Whether the factorial decomposition proocedure converged or not. 
		:KmConverged: Whether the K-means proocedure converged or not. 

		:TSS: Total sum of squared deviations for best loop.
		:BSS: Between sum of squared deviations for best loop.
		:RSS: Residual sum of squared deviations for best loop.
		:PseudoF: PsuedoF score from the best loop.
	
		:Labels: Cluster labels for the best loop.

		:Fs_km: K-means clustering objective function values.
		:Fs_fa: Factor decomposition objective function values.
		:Enorm: Frobenius or L2 norm of residual term from the model.'''

# === REFERENCES
_doc_refs = '''
	References:
		[1] Vichi, Maurizio & Rocci, Roberto & Kiers, Henk. (2007).
		Simultaneous Component and Clustering Models for Three-way Data: Within and Between Approaches.
		Journal of Classification. 24. 71-98. 10.1007/s00357-007-0006-x. 

		[2] Bro, R. (1998).
		Multi-way analysis in the food industry: models, algorithms, and applications.
		Universiteit van Amsterdam.
		
		[3] P. Arabie and L. Hubert.
		Advances in cluster analysis relevant to marketing research.
		In Wolfgang Gaul and Dietmar Pfeifer, editors, From Data to Knowledge, pages 3–19,
		Berlin, Heidelberg, 1996. Springer Berlin Heidelberg.
		
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


# === The TWCFTA MODEL
@_doc_formatter(_doc_init_args)
class TWCFTA(_BaseClass):
	"""
	Three-way Clustering-Factorial Tandem model (TWCFTA).
		- Perform KMeans on X_i_jk to obtain membership matrix U_i_g and centroids X_g_jk in full space.
		- Obtain Y_g_qr and C_k_r & B_j_q factor matrices that maximizes the reconstruction of U_i_g.X_g_jk via Tucker2.
	{0}
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
	@_doc_formatter(_doc_fit_args, _doc_init_attrs, _doc_refs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		"""
		{0}
		{1}
		{2}
		"""

		# initializing basic config
		rng = np.random.default_rng(self.random_state)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_params()
		self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_i_jk = (X_i_jk - X_i_jk.mean(axis=0, keepdims=True))/X_i_jk.std(axis=0, keepdims=True)

		headers = ['Loop','KM Iter','FA Iter','Loop time','BSS (%)','PseudoF','KM_Converged','FA_Converged']
		if self.verbose: print(tabulate([],headers=headers))

		# Factorial reduction on centroids (via T2 applied to the centroids matrix X_g_jk_bar)
		for loop in range(1, self.n_loops+1):
			
			start_time = time()

			if self.init == 'svd':
				if loop == 2: break

			# ------------ Start KMeans Clustering ------------
			
			# given directly as paramters
			U_i_g0 = self.U_i_g
			km_iter = 0
			km_converged = False
			Fs_km = []

			if U_i_g0 is None:

				U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)
				U_i_g_init = U_i_g0.copy()

				# initial objective
				X_g_jk0 = np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T @ X_i_jk  # compute centroids matrix
				F0 = np.linalg.norm(U_i_g0@X_g_jk0,2)  # residual matrix
				Fs_km.append(F0)

				# clustering on objects (via KMeans applied to X_i_jk)
				conv = 2*self.tol
				
				# print('initial',U_i_g0)
				while conv > self.tol:
					
					km_iter += 1

					# get random centroids
					U_i_g = OneKMeans(X_i_jk, G, U_i_g=U_i_g0, rng=rng)  # updated membership matrix
					X_g_jk = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ X_i_jk  # compute centroids matrix
					
					# check if maximizes orbjective or minimizes the loss
					F = np.linalg.norm(U_i_g@X_g_jk,2)  # residual matrix
					conv = abs(F-Fs_km[-1])
					Fs_km.append(F)

					if km_iter == self.n_max_iter:
						if self.verbose: print("KM Maximum iterations reached.")
						break

					if conv < self.tol:
						km_converged = True
					else:
						F0 = F
						U_i_g0 = U_i_g
			else:
				U_i_g_init = np.array(U_i_g0)
				U_i_g = U_i_g_init
			# print('final',U_i_g)
			
			# updated centroids
			X_g_jk = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ X_i_jk  # compute centroids matrix			

			# ------------ End KMeans Clustering ------------
			
			# ------------ Start of initialization for FA ------------

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
				if B_j_q0 is None: B_j_q0 = SingularVectors(X_j_kg@X_j_kg.T, Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(X_k_gj@X_k_gj.T, R)
			else:  # random initialization
				if B_j_q0 is None: B_j_q0 = SingularVectors(rng.random([J,J]), Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(rng.random([K,K]), R)
			
			I_g_g = np.eye(G)

			# ----------- End of initialization --------------

			# ----------- Start of objective function --------------
			
			B_j_q_init = B_j_q0
			C_k_r_init = C_k_r0

			# updated centroids matrix
			Y_g_qr = X_g_jk @ np.kron(C_k_r0, B_j_q0)

			F0 = np.linalg.norm(Y_g_qr,2)
			conv = 2*self.tol
			Fs_fa.append(F0)

			# ----------- End of Objective Function Definition --------------

			while (conv > self.tol):

				fa_iter += 1

				# ----------- Start of factor matrices update --------------

				# updating B_j_q
				B_j_j = X_j_kg @ np.kron(I_g_g, C_k_r0@C_k_r0.T) @ X_j_kg.T
				B_j_q = SingularVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_gj @ np.kron(B_j_q@B_j_q.T, I_g_g) @ X_k_gj.T
				C_k_r = SingularVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------

				# ----------- Start of objective functions update --------------

				# updated centroids matrix
				Y_g_qr = X_g_jk @ np.kron(C_k_r, B_j_q)

				# compute L2 norm of reconstruction error
				F = np.linalg.norm(Y_g_qr,2)
				conv = abs(F-Fs_fa[-1])
				Fs_fa.append(F)

				# ----------- End of objective functions update --------------

				# ----------- Start of convergence test --------------

				if (fa_iter == self.n_max_iter):
					if self.verbose: print("FA Maximum iterations reached.")
					break

				if (conv < self.tol):
					fa_converged = True
				else:
					F0 = F
					B_j_q0 = B_j_q
					C_k_r0 = C_k_r

				# ----------- End of convergence test --------------

			# ----------- Start of results update for each loop --------------

			time_elapsed = time()-start_time

			Y_i_qr = X_i_jk @ np.kron(C_k_r, B_j_q)
			Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr
			Z_i_qr = U_i_g @ Y_g_qr

			TSS = Y_i_qr.var()*Y_i_qr.size
			BSS = Z_i_qr.var()*Z_i_qr.size  # clustering + factor reduction
			RSS = (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size  # redsidual sum of squares 

			BSS_percent = (BSS/TSS)*100  # between cluster deviance

			# pseudoF and output results
			pseudoF = round(PseudoF(BSS, RSS, full_tensor_shape, reduced_tensor_shape), 4) if G not in [1,I] else None
			if self.verbose: print(tabulate([[]], headers=[loop, km_iter, fa_iter, round(time_elapsed,4), round(BSS_percent,4), pseudoF, km_converged, fa_converged], tablefmt='plain'))

			if (loop == 1):
				B_j_q_simu = B_j_q
				C_k_r_simu = C_k_r
				U_i_g_simu = U_i_g
				Y_g_qr_simu = Y_g_qr
				km_iter_simu = km_iter
				fa_iter_simu = fa_iter
				loop_simu = 1
				km_converged_simu = km_converged
				fa_converged_simu = fa_converged
				BSS_per = BSS_percent
				TSS_simu = TSS
				BSS_simu = BSS
				RSS_simu = RSS
				Fs_fa = Fs_fa
				Fs_km = Fs_km
				pseudoF_simu = pseudoF
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			if (BSS_percent > BSS_per):
				B_j_q_simu = B_j_q
				C_k_r_simu = C_k_r
				U_i_g_simu = U_i_g
				Y_g_qr_simu = Y_g_qr
				km_iter_simu = km_iter  # number of iterations until convergence
				fa_iter_simu = fa_iter
				loop_simu = loop  # best loop so far
				km_converged_simu = km_converged  # if there was a convergence
				fa_converged_simu = fa_converged  # if there was a convergence
				BSS_per = BSS_percent  # best objective functional value
				TSS_simu = TSS
				BSS_simu = BSS
				RSS_simu = RSS
				Fs_fa = Fs_fa  # objective function values for FA
				Fs_km = Fs_km
				pseudoF_simu = pseudoF
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			# ----------- End of results update for each loop --------------

		# ----------- Start of result update for best loop --------------

		Y_i_qr = X_i_jk @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g)@U_i_g.T@Y_i_qr
		Z_i_qr = U_i_g_simu @ Y_g_qr

		# factor matrices and centroid matrices
		self.U_i_g = U_i_g_simu
		self.U_i_g0 = U_i_g_init_simu
		self.B_j_q0 = B_j_q_init_simu
		self.C_k_r0 = C_k_r_init_simu
		self.B_j_q = B_j_q_simu
		self.C_k_r = C_k_r_simu
		self.Y_g_qr = Y_g_qr
		self.X_i_jk_scaled = X_i_jk

		# total time taken
		self.BestTimeElapsed = best_time_elapsed_simu
		self.BestLoop = loop_simu
		self.BestKMIteration = km_iter_simu
		self.BestFAIteration = fa_iter_simu
		self.FaConverged = fa_converged_simu
		self.KmConverged = km_converged_simu

		# maximum between cluster deviance
		self.TSS = TSS_simu
		self.BSS = BSS_simu
		self.RSS = RSS_simu
		self.PseudoF = pseudoF_simu

		# Error in model
		self.Enorm = 1/I*np.linalg.norm(X_i_jk - U_i_g_simu @ Y_g_qr @ np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.Fs_km = Fs_km  # objective values for kmeans
		self.Fs_fa = Fs_fa  # objective values for factor decomposition

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self


# === The TWFCTA MODEL
@_doc_formatter(_doc_init_args)
class TWFCTA(_BaseClass):
	"""
	Three-way Factorial-Clustering Tandem model (TWFCTA).
		- Apply Factorial Decomposition (via Tucker2 applied to X_i_jk) to obtain C_k_r and B_j_q component factors.
		- Then perform Clustering (via KMeans applied to Y_i_qr) to obtain U_i_g and Y_g_qr.
	
	:param X_i_jk: (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
	:param full_tensor_shape: (I,J,K) dimensions of the original tensor.
	:param reduced_tensor_shape: (G,Q,R) dimensions of centroids tensor.

	{0}
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

	@_doc_formatter(_doc_init_attrs, _doc_refs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape):
		"""
		:Parameters:
			- X_i_jk: (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
			- full_tensor_shape: (I,J,K) dimensions of the original tensor.
			- reduced_tensor_shape: (G,Q,R) dimensions of centroids tensor.
		
		{0}
		{1}
		"""

		# initializing basic config
		rng = np.random.default_rng(self.random_state)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_params()
		self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)

		# scaling X_i_jk
		X_i_jk = (X_i_jk - X_i_jk.mean(axis=0, keepdims=True))/X_i_jk.std(axis=0, keepdims=True)

		# declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# matricize centroid tensor
		X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
		X_j_ki = Unfold(X_k_i_j, mode=2)
		X_k_ij = Unfold(X_k_i_j, mode=0)

		I_i_i = np.diag(np.ones(I))  # identity matrix

		headers = ['Loop','KM Iter','FA Iter','Loop time','BSS (%)','PseudoF','KM_Converged','FA_Converged']
		if self.verbose == True: print(tabulate([],headers=headers))

		# number of loops
		for loop in range(1, self.n_loops+1):
			
			start_time = time()

			if self.init == 'svd':
				if loop == 2: break

			# ------------ Start of initialization for FA ------------

			fa_iter = 0
			Fs_fa = []  # objective function values
			converged = False

			# as direct input
			B_j_q0 = self.B_j_q
			C_k_r0 = self.C_k_r

			# initialize B and C
			if self.init == 'svd':
				if B_j_q0 is None: B_j_q0 = SingularVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(X_k_ij@X_k_ij.T, R)
			else:  # random initialization
				if B_j_q0 is None: B_j_q0 = rng.random([J,Q])
				if C_k_r0 is None: C_k_r0 = rng.random([K,R])

			# ----------- End of initialization --------------

			# ----------- Start of objective function --------------

			B_j_q_init = B_j_q0
			C_k_r_init = C_k_r0

			# updated centroids matrix
			Y_i_qr = X_i_jk @ np.kron(C_k_r0, B_j_q0)

			F0 = np.linalg.norm(Y_i_qr,2)
			conv = 2*self.tol
			Fs_fa.append(F0)
			fa_converged = False

			# ----------- End of Objective Function Definition --------------

			while (conv > self.tol):

				fa_iter += 1

				# ----------- Start of factor matrices update --------------

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(I_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = SingularVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, I_i_i) @ X_k_ij.T
				C_k_r = SingularVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------

				# ----------- Start of objective functions update --------------

				# updated centroids matrix
				Y_i_qr = X_i_jk @ np.kron(C_k_r, B_j_q)

				# compute L2 norm of reconstruction error
				F = np.linalg.norm(Y_i_qr, 2)
				conv = abs(F-Fs_fa[-1])
				Fs_fa.append(F)

				# ----------- End of objective functions update --------------

				# ----------- Start stopping criteria check --------------

				if (fa_iter == self.n_max_iter):
					if self.verbose == True: print("FA Maximum iterations reached.")
					break

				if (conv < self.tol):
					fa_converged = True
				else:
					F0 = F
					B_j_q0 = B_j_q
					C_k_r0 = C_k_r

				# ----------- End stopping criteria check --------------

			# ----------- Start KMeans clustering applied to Y_i_qr --------------

			Y_i_qr = X_i_jk @ np.kron(C_k_r, B_j_q)
			km_iter = 0
			km_converged = False
			Fs_km = []

			# given directly as paramters
			U_i_g0 = self.U_i_g
			if U_i_g0 is None:
				U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)
				U_i_g_init = U_i_g0.copy()

				# initial objective
				Y_g_qr0 = np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T @ Y_i_qr  # compute centroids matrix
				F0 = np.linalg.norm(U_i_g0@Y_g_qr0,2)  # residual matrix
				Fs_km.append(F0)

				# clustering on objects (via KMeans applied to X_i_jk)
				conv = 2*self.tol
				
				# print('initial',U_i_g0)

				while conv > self.tol:
					
					km_iter += 1

					# get random centroids
					U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, rng=rng)  # updated membership matrix
					Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr  # compute centroids matrix
					
					# check if maximizes orbjective or minimizes the loss
					F = np.linalg.norm(U_i_g@Y_g_qr, 2)  # residual matrix
					conv = abs(F-Fs_km[-1])
					Fs_km.append(F)

					# not converged
					if km_iter == self.n_max_iter:
						if self.verbose == True: print("KM Maximum iterations reached.")
						break
					
					if conv < self.tol:
						km_converged = True
					else:
						F0 = F
						U_i_g0 = U_i_g
			else:
				U_i_g_init = U_i_g0.copy()
				U_i_g = U_i_g_init

			# print('final',U_i_g)
			
			# ----------- End KMeans clustering applied to Y_i_qr --------------

			# ----------- Start of results update for each loop --------------

			time_elapsed = time()-start_time

			Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ X_i_jk @ np.kron(C_k_r, B_j_q)
			Z_i_qr = U_i_g @ Y_g_qr

			TSS = Y_i_qr.var()*Y_i_qr.size
			BSS = Z_i_qr.var()*Z_i_qr.size  # clustering + factor reduction
			RSS = (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size  # redsidual sum of squares 

			BSS_percent = (BSS/TSS)*100  # between cluster deviance

			# pseudoF and output results
			pseudoF = round(PseudoF(BSS, RSS, full_tensor_shape, reduced_tensor_shape), 4) if G not in [1,I] else None
			if self.verbose: print(tabulate([[]], headers=[loop, km_iter, fa_iter, round(time_elapsed,4), round(BSS_percent,4), pseudoF, km_converged, fa_converged], tablefmt='plain'))

			if (loop == 1):
				B_j_q_simu = B_j_q
				C_k_r_simu = C_k_r
				U_i_g_simu = U_i_g
				Y_g_qr_simu = Y_g_qr
				km_iter_simu = km_iter
				fa_iter_simu = fa_iter
				loop_simu = 1
				km_converged_simu = km_converged
				fa_converged_simu = fa_converged
				BSS_per_simu = BSS_percent
				TSS_simu = TSS
				BSS_simu = BSS
				RSS_simu = RSS
				Fs_fa = Fs_fa
				Fs_km = Fs_km
				pseudoF_simu = pseudoF
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed


			if (BSS_percent > BSS_per_simu):
				B_j_q_simu = B_j_q
				C_k_r_simu = C_k_r
				U_i_g_simu = U_i_g
				Y_g_qr_simu = Y_g_qr
				km_iter_simu = km_iter  # number of iterations until convergence
				fa_iter_simu = fa_iter
				loop_simu = loop  # best loop so far
				km_converged_simu = km_converged  # if there was a convergence
				fa_converged_simu = fa_converged  # if there was a convergence
				converged_simu = converged  # if there was a convergence
				BSS_per_simu = BSS_percent  # best objective functional value
				TSS_simu = TSS
				BSS_simu = BSS
				RSS_simu = RSS
				Fs_fa = Fs_fa  # objective function values for FA
				Fs_km = Fs_km
				pseudoF_simu = pseudoF
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			# ----------- End of results update for each loop --------------

		# ----------- Start of result update for best loop --------------

		Y_i_qr = X_i_jk @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.linalg.inv(U_i_g.T@U_i_g)@U_i_g.T@Y_i_qr
		Z_i_qr = U_i_g_simu @ Y_g_qr

		# factor matrices and centroid matrices
		self.U_i_g = U_i_g_simu
		self.U_i_g0 = U_i_g_init_simu
		self.B_j_q0 = B_j_q_init_simu
		self.C_k_r0 = C_k_r_init_simu
		self.B_j_q = B_j_q_simu
		self.C_k_r = C_k_r_simu
		self.Y_g_qr = Y_g_qr_simu
		self.X_i_jk_scaled = X_i_jk

		# total time taken
		self.BestTimeElapsed = best_time_elapsed_simu
		self.BestLoop = loop_simu
		self.BestKMIteration = km_iter_simu
		self.BestFAIteration = fa_iter_simu
		self.FaConverged = fa_converged_simu
		self.KmConverged = km_converged_simu

		# maximum between cluster deviance
		self.TSS = TSS_simu
		self.BSS = BSS_simu
		self.RSS = RSS_simu
		self.PseudoF = pseudoF_simu

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# Error in model
		self.Fs_km = Fs_km  # all error norms
		self.Fs_fa = Fs_fa  # all error norms
		self.Enorm = 1/I*np.linalg.norm(X_i_jk - U_i_g_simu@Y_g_qr_simu @ np.kron(C_k_r_simu, B_j_q_simu).T, 2)

		# ----------- End of result update for best loop --------------

		return self
