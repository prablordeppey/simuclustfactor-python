### ---- IMPORTING MODULES

import numpy as np
from .tensor import Fold, Unfold
from .utils import SingularVectors, PseudoF, _BaseClass, OneKMeans, RandomMembershipMatrix
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
		random_state (boolean): seed for random generations. Defaults to None.
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

		- TSSFull: Total sum of squared deviations for best loop in the full space.
		- BSSFull: Between sum of squared deviations for best loop in the full space.
		- RSSFull: Residual sum of squared deviations for best loop in the full space.
		- PFFull: PsuedoF score from the best loop in the full space.

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
@_doc_formatter(_doc_init_args, _doc_init_attrs)
class T3Clus(_BaseClass):
	"""
	Simultaneous version of TWCFTA (T3Clus).
	Maximizes the between cluster deviance of the component scores.
	{0}
	{1}
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

	@_doc_formatter(_doc_fit_args, _doc_init_attrs, _doc_refs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape, U_i_g=None, C_k_r=None, twcfta_init='random', twfcta_init='random'):
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
		self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_i_jk = (X_i_jk - X_i_jk.mean(axis=0, keepdims=True))/X_i_jk.std(axis=0, keepdims=True)
		
		# permuting X_i_jk by mode
		X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
		X_k_ij = Unfold(X_k_i_j, mode=0)
		X_j_ki = Unfold(X_k_i_j, mode=2)

		I_jk_jk = np.diag(np.ones(J*K))
		I_i_i = np.diag(np.ones(I))

		headers = ['Loop','Best Iteration','Time Elapsed','BSS Full (%)', 'BSS Reduced (%)', 'PseudoF Full', 'PseudoF Reduced','Convergence']
		if self.verbose: print(tabulate([],headers=headers))
		
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
				if B_j_q0 is None: B_j_q0 = SingularVectors(rng.random([J,J]), Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(rng.random([K,K]), R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			elif self.init == 'twcfta':
				cft = TWCFTA(random_state=self.random_state, init=twcfta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = cft.B_j_q
				if C_k_r0 is None: C_k_r0 = cft.C_k_r
				if U_i_g0 is None: U_i_g0 = cft.U_i_g
			
			elif self.init == 'twfcta':
				fct = TWFCTA(random_state=self.random_state, init=twfcta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = fct.B_j_q
				if C_k_r0 is None: C_k_r0 = fct.C_k_r
				if U_i_g0 is None: U_i_g0 = fct.U_i_g
			
			else:  # svd initialization

				if loop==2: break  # run once if not random

				# initializing B, C
				if B_j_q0 is None: B_j_q0 = SingularVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(X_k_ij@X_k_ij.T, R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# ----------- Start of Objective Function Definition --------------

			U_i_g_init = U_i_g0.copy()
			B_j_q_init = B_j_q0.copy()
			C_k_r_init = C_k_r0.copy()

			Z_i_qr = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T @ X_i_jk @ np.kron(C_k_r0, B_j_q0) # model
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

				Hu_i_i = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(Hu_i_i-I_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = SingularVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, Hu_i_i-I_i_i) @ X_k_ij.T
				C_k_r = SingularVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------
				
				# ----------- Start of objects membership matrix update --------------

				Y_i_qr = X_i_jk @ np.kron(C_k_r, B_j_q) # component scores
				U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, seed=rng)  # updated membership matrix

				# ----------- End of objects membership matrix update --------------

				# ----------- Start of objective functions update --------------

				Z_i_qr = U_i_g @ np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr @ np.kron(C_k_r, B_j_q).T

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
					break

				if iteration == self.n_max_iter:
					if self.verbose: print("Maximum iterations reached.")
					break

				# ----------- End stopping criteria check --------------

				U_i_g0 = U_i_g
				B_j_q0 = B_j_q
				C_k_r0 = C_k_r

			# ----------- Start of results update for each loop --------------

			time_elapsed = time()-start_time

			# updating X
			Y_i_qr = X_i_jk @ np.kron(best_C_k_r, best_B_j_q)
			Y_g_qr = np.linalg.inv(best_U_i_g.T@best_U_i_g) @ best_U_i_g.T @ Y_i_qr
			Z_i_qr = best_U_i_g @  Y_g_qr
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T

			TSS_full = X_i_jk.var()*X_i_jk.size
			BSS_full = Z_i_jk.var()*Z_i_jk.size
			RSS_full = (X_i_jk-Z_i_jk).var()*(X_i_jk-Z_i_jk).size

			TSS_reduced = Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size

			BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
			BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
			pseudoF_full = round(PseudoF(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			pseudoF_reduced = round(PseudoF(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			best_PF = pseudoF_full

			# output results
			if self.verbose: print(tabulate([[]], headers=[loop, best_iteration, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), pseudoF_full, pseudoF_reduced, converged], tablefmt='plain'))

			# tracking the best loop iterates
			if (loop == 1):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = 1
				converged_simu = converged
				Fs_simu = Fs
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			if (best_PF > best_PF_simu):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration  # number of iterations until convergence
				loop_simu = loop  # best loop so far
				converged_simu = converged  # if there was a convergence
				Fs_simu = Fs  # objective function values
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			# ----------- End of results update for each loop --------------

		# ----------- Start of result update for best loop --------------

		Y_i_qr = X_i_jk @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.linalg.inv(U_i_g_simu.T@U_i_g_simu) @ U_i_g_simu.T@Y_i_qr
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
		self.BestIteration = iteration_simu

		# maximum between cluster deviance
		self.TSSFull = TSS_full_simu
		self.BSSFull = BSS_full_simu
		self.RSSFull = RSS_full_simu
		self.TSSReduced = TSS_reduced_simu
		self.BSSReduced = BSS_reduced_simu
		self.RSSReduced = RSS_reduced_simu
		self.PFFull = pseudoF_full
		self.PFReduced = pseudoF_reduced

		# Error in model
		self.Enorm = 1/I*np.linalg.norm(X_i_jk - Z_i_qr @ np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.Fs = Fs_simu  # all error norms

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self



# === The TFKMeans MODEL
@_doc_formatter(_doc_init_args, _doc_init_attrs)
class TFKMeans(_BaseClass):
	"""
	Simultaneous version of TWFCTA - Clustering-Factorial Decomposition (3FKMeans).
	Minimize within cluster deviance of the component scores.
	{0}
	{1}
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


	@_doc_formatter(_doc_fit_args, _doc_init_attrs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape, B_j_q=None, C_k_r=None, twcfta_init='random', twfcta_init='random'):
		'''
		{0}
		{1}
		'''

		rng = np.random.default_rng(self.random_state)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_i_jk = (X_i_jk - X_i_jk.mean(axis=0, keepdims=True))/X_i_jk.std(axis=0, keepdims=True)

		I_jk_jk = np.diag(np.ones(J*K))
		I_i_i = np.diag(np.ones(I))

		headers = ['Loop','Best Iteration','Time Elapsed','BSS Full (%)','BSS Reduced (%)', 'PseudoF Full', 'PseudoF Reduced', 'Convergence']
		if self.verbose: print(tabulate([],headers=headers))
		
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
				if B_j_q0 is None: B_j_q0 = SingularVectors(rng.random([J,J]), Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(rng.random([K,K]), R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# if loop==2: break  # run once if not random
			elif self.init == 'twcfta':
				cft = TWCFTA(random_state=self.random_state, init=twcfta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = cft.B_j_q
				if C_k_r0 is None: C_k_r0 = cft.C_k_r
				if U_i_g0 is None: U_i_g0 = cft.U_i_g
			
			elif self.init == 'twfcta':
				fct = TWFCTA(random_state=self.random_state, init=twfcta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = fct.B_j_q
				if C_k_r0 is None: C_k_r0 = fct.C_k_r
				if U_i_g0 is None: U_i_g0 = fct.U_i_g
			
			else:  # svd initialization
				
				if loop==2: break  # run once if not random

				# permuting X_i_jk by mode
				X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
				X_j_ki = Unfold(X_k_i_j, mode=2)
				X_k_ij = Unfold(X_k_i_j, mode=0)

				# initializing B, C
				if B_j_q0 is None: B_j_q0 = SingularVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(X_k_ij@X_k_ij.T, R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# ----------- Start of Objective Function Definition --------------

			U_i_g_init = U_i_g0.copy()
			B_j_q_init = B_j_q0.copy()
			C_k_r_init = C_k_r0.copy()

			# updating X_i_jk
			X_i_jk_N = X_i_jk @ np.kron(C_k_r0@C_k_r0.T, B_j_q0@B_j_q0.T)
			Z_i_qr = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T @ X_i_jk_N @ np.kron(C_k_r0, B_j_q0)  # updated centroids matrix

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

				Hu_i_i = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T

				# permuting X_i_jk_N by mode
				X_k_i_j = Fold(X_i_jk_N, mode=1, shape=(K,I,J))
				X_k_ij = Unfold(X_k_i_j, mode=0)
				X_j_ki = Unfold(X_k_i_j, mode=2)

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(Hu_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = SingularVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, Hu_i_i) @ X_k_ij.T
				C_k_r = SingularVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------
				# # updating X
				
				# ----------- Start of objects membership matrix update --------------

				X_i_jk_N = X_i_jk @ np.kron(C_k_r @ C_k_r.T, B_j_q @ B_j_q.T)
				Y_i_qr = X_i_jk_N @ np.kron(C_k_r, B_j_q) # component scores
				U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, seed=self.random_state)  # updated membership matrix

				# ----------- End of objects membership matrix update --------------

				# ----------- Start of objective functions update --------------

				Z_i_qr = U_i_g @ np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr

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
					break

				if iteration == self.n_max_iter:
					if self.verbose: print("Maximum iterations reached.")
					break				

				# ----------- End stopping criteria check --------------

				U_i_g0 = U_i_g
				B_j_q0 = B_j_q
				C_k_r0 = C_k_r

			# ----------- Start of results update for each loop --------------

			time_elapsed = time()-start_time

			# updating X
			X_i_jk_N = X_i_jk @ np.kron(best_C_k_r@best_C_k_r.T, best_B_j_q@best_B_j_q.T)
			Y_i_qr = X_i_jk_N @ np.kron(best_C_k_r, best_B_j_q)  # (PxQ) object centroid matrix.
			Y_g_qr = np.linalg.inv(best_U_i_g.T@best_U_i_g) @ best_U_i_g.T @ Y_i_qr
			Z_i_qr = best_U_i_g @  Y_g_qr
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T

			TSS_full = X_i_jk.var()*X_i_jk.size
			BSS_full = Z_i_jk.var()*Z_i_jk.size
			RSS_full = (X_i_jk-Z_i_jk).var()*(X_i_jk-Z_i_jk).size

			TSS_reduced = Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size

			BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
			BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
			pseudoF_full = round(PseudoF(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			pseudoF_reduced = round(PseudoF(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			best_PF = pseudoF_reduced

			# output results
			if self.verbose: print(tabulate([[]], headers=[loop, best_iteration, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), pseudoF_full, pseudoF_reduced, converged], tablefmt='plain'))

			# tracking the best loop iterates			
			if (loop == 1):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = 1
				converged_simu = converged
				Fs_simu = Fs
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			if (best_PF > best_PF_simu):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = loop
				converged_simu = converged				
				Fs_simu = Fs
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			# ----------- End of results update for each loop --------------

		# ----------- Start of result update for best loop --------------

		X_i_jk_N = X_i_jk @ np.kron(C_k_r_simu@C_k_r_simu.T, B_j_q_simu@B_j_q_simu.T)
		
		Y_i_qr = X_i_jk_N @ np.kron(C_k_r_simu, B_j_q_simu)
		Y_g_qr = np.linalg.inv(U_i_g_simu.T@U_i_g_simu)@U_i_g.T@Y_i_qr
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

		# maximum between cluster deviance
		self.TSSFull = TSS_full_simu
		self.BSSFull = BSS_full_simu
		self.RSSFull = RSS_full_simu
		self.TSSReduced = TSS_reduced_simu
		self.BSSReduced = BSS_reduced_simu
		self.RSSReduced = RSS_reduced_simu
		self.PFFull = pseudoF_full
		self.PFReduced = pseudoF_reduced

		# Error in model
		self.Enorm = 1/I*np.linalg.norm(X_i_jk_N - Z_i_qr@np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.Fs = Fs_simu  # all error norms
		self.converged = converged_simu

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self


# === The CT3CLUS MODEL
@_doc_formatter(_doc_init_args, _doc_init_attrs)
class CT3Clus(_BaseClass):
	"""
	Combination of T3Clus and TFKMeans.
	{0}
	{1}
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

	@_doc_formatter(_doc_fit_args, _doc_init_attrs, _doc_refs)
	def fit(self, X_i_jk, full_tensor_shape, reduced_tensor_shape, alpha=0.5, twcfta_init='random', twfcta_init='random'):
		'''
		Fit Args:
			:X_i_jk (ndarray): (I,JK) mode-1 matricized three-way arrays with frontal slabs next to each other. It is column centered.
			:full_tensor_shape (tuple): (I,J,K) dimensions of the original tensor.
			:reduced_tensor_shape (tuple): (G,Q,R) dimensions of centroid tensor.
			:alpha (float): 0<=alpha<=1. alpha=1 for T3Clus, alpha=0 for TFKMeans. alpha=0.5 improve recoverability.
		{1}
		{2}
		'''
		# initializing basic config
		rng = np.random.default_rng(self.random_state)  # random number generator
		self.reduced_tensor_shape = reduced_tensor_shape  # (I,J,K) tensor shape
		self.full_tensor_shape = full_tensor_shape  # (G,Q,R) core tensor shape

		# check parameters and arguments
		self._check_ct3clus_params(full_tensor_shape, reduced_tensor_shape, alpha)
		self._check_initialized_components()
		X_i_jk = np.array(X_i_jk)
		
		# Declaring I,J,K and G,Q,R
		I,J,K = full_tensor_shape
		G,Q,R = reduced_tensor_shape

		# standardizing the dataset X_i_jk
		X_i_jk = (X_i_jk - X_i_jk.mean(axis=0, keepdims=True))/X_i_jk.std(axis=0, keepdims=True)

		I_jk_jk = np.diag(np.ones(J*K))
		I_i_i = np.diag(np.ones(I))

		headers = ['Loop','Iteration','Time Elapsed','BSS Full (%)', 'BSS Reduced (%)', 'PseudoF Full', 'PseudoF Reduced','Convergence']
		if self.verbose: print(tabulate([],headers=headers))
		
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
				if B_j_q0 is None: B_j_q0 = SingularVectors(rng.random([J,J]), Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(rng.random([K,K]), R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# run once if not random
			elif self.init == 'twcfta':
				cft = TWCFTA(random_state=self.random_state, init=twcfta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = cft.B_j_q
				if C_k_r0 is None: C_k_r0 = cft.C_k_r
				if U_i_g0 is None: U_i_g0 = cft.U_i_g
			
			elif self.init == 'twfcta':
				fct = TWFCTA(random_state=self.random_state, init=twfcta_init).fit(X_i_jk, full_tensor_shape, reduced_tensor_shape)
				if B_j_q0 is None: B_j_q0 = fct.B_j_q
				if C_k_r0 is None: C_k_r0 = fct.C_k_r
				if U_i_g0 is None: U_i_g0 = fct.U_i_g
			
			else:  # svd initialization
				
				if loop==2: break  # run once if not random

				# permuting X_i_jk by mode
				X_k_i_j = Fold(X_i_jk, mode=1, shape=(K,I,J))
				X_j_ki = Unfold(X_k_i_j, mode=2)
				X_k_ij = Unfold(X_k_i_j, mode=0)

				# initializing B, C
				if B_j_q0 is None: B_j_q0 = SingularVectors(X_j_ki@X_j_ki.T, Q)
				if C_k_r0 is None: C_k_r0 = SingularVectors(X_k_ij@X_k_ij.T, R)
				if U_i_g0 is None: U_i_g0 = RandomMembershipMatrix(I, G, rng=rng)

			# ----------- Start of Objective Function Definition --------------

			U_i_g_init = U_i_g0.copy()
			B_j_q_init = B_j_q0.copy()
			C_k_r_init = C_k_r0.copy()

			# updating X_i_jk
			P = np.kron(C_k_r0 @ C_k_r0.T, B_j_q0 @ B_j_q0.T)
			X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )

			Z_i_qr = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T @ X_i_jk_N @ np.kron(C_k_r0, B_j_q0)  # (PxQ) object centroid matrix. It identifies the P centroids in the reduced space of the principal components.

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

				Hu_i_i = U_i_g0 @ np.linalg.inv(U_i_g0.T@U_i_g0) @ U_i_g0.T
				# Hu_i_i = U_i_g0 @ np.diag(1/U_i_g0.sum(axis=0)) @ U_i_g0.T

				# permuting X_i_jk_N by mode
				X_k_i_j = Fold(X_i_jk_N, mode=1, shape=(K,I,J))
				X_k_ij = Unfold(X_k_i_j, mode=0)
				X_j_ki = Unfold(X_k_i_j, mode=2)

				# updating B_j_q
				B_j_j = X_j_ki @ np.kron(Hu_i_i-alpha*I_i_i, C_k_r0@C_k_r0.T) @ X_j_ki.T
				B_j_q = SingularVectors(B_j_j, Q)

				# updating C_k_r
				C_k_k = X_k_ij @ np.kron(B_j_q@B_j_q.T, Hu_i_i-alpha*I_i_i) @ X_k_ij.T
				C_k_r = SingularVectors(C_k_k, R)

				# ----------- End of factor matrices update --------------
	
				# ----------- Start of objects membership matrix update --------------

				# updating X
				P = np.kron(C_k_r @ C_k_r.T, B_j_q @ B_j_q.T)
				X_i_jk_N = X_i_jk @ (P + (alpha**0.5)*(I_jk_jk-P) )

				Y_i_qr = X_i_jk_N @ np.kron(C_k_r, B_j_q) # component scores
				U_i_g = OneKMeans(Y_i_qr, G, U_i_g=U_i_g0, seed=self.random_state)  # updated membership matrix

				# ----------- End of objects membership matrix update --------------

				# ----------- Start of objective functions update --------------

				Z_i_qr = U_i_g @ np.linalg.inv(U_i_g.T@U_i_g) @ U_i_g.T @ Y_i_qr

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
					break

				if iteration == self.n_max_iter:
					if self.verbose: print("Maximum iterations reached.")
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
			Y_i_qr = X_i_jk_N @ np.kron(best_C_k_r, best_B_j_q)  # (PxQ) object centroid matrix.
			Y_g_qr = np.linalg.inv(best_U_i_g.T@best_U_i_g) @ best_U_i_g.T @ Y_i_qr
			Z_i_qr = best_U_i_g @  Y_g_qr
			Z_i_jk = Z_i_qr @ np.kron(best_C_k_r, best_B_j_q).T

			TSS_full = X_i_jk.var()*X_i_jk.size
			BSS_full = Z_i_jk.var()*Z_i_jk.size
			RSS_full = (X_i_jk-Z_i_jk).var()*(X_i_jk-Z_i_jk).size

			TSS_reduced = Y_i_qr.var()*Y_i_qr.size
			BSS_reduced = Z_i_qr.var()*Z_i_qr.size
			RSS_reduced = (Y_i_qr-Z_i_qr).var()*(Y_i_qr-Z_i_qr).size

			BSS_percent_full = (BSS_full/TSS_full)*100  # between cluster deviance
			BSS_percent_reduced = (BSS_reduced/TSS_reduced)*100  # between cluster deviance
			pseudoF_full = round(PseudoF(BSS_full, RSS_full, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None
			pseudoF_reduced = round(PseudoF(BSS_reduced, RSS_reduced, full_tensor_shape, reduced_tensor_shape),4) if G not in [1,I] else None

			best_PF = alpha*pseudoF_full + (1-alpha)*pseudoF_reduced

			# output results
			if self.verbose: print(tabulate([[]], headers=[loop, best_iteration, round(time_elapsed,4), round(BSS_percent_full,2), round(BSS_percent_reduced,2), pseudoF_full, pseudoF_reduced, converged], tablefmt='plain'))

			# tracking the best loop iterates
			if (loop == 1):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = 1
				converged_simu = converged
				Fs_simu = Fs
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
				TSS_reduced_simu = TSS_reduced
				BSS_reduced_simu = BSS_reduced
				RSS_reduced_simu = RSS_reduced
				U_i_g_init_simu = U_i_g_init
				B_j_q_init_simu = B_j_q_init
				C_k_r_init_simu = C_k_r_init
				best_time_elapsed_simu = time_elapsed

			if (best_PF > best_PF_simu):
				B_j_q_simu = best_B_j_q
				C_k_r_simu = best_C_k_r
				U_i_g_simu = best_U_i_g
				iteration_simu = best_iteration
				loop_simu = loop
				converged_simu = converged				
				Fs_simu = Fs
				pseudoF_full_simu = pseudoF_full
				pseudoF_reduced_simu = pseudoF_reduced
				best_PF_simu = best_PF
				TSS_full_simu = TSS_full
				BSS_full_simu = BSS_full
				RSS_full_simu = RSS_full
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
		Y_g_qr = np.linalg.inv(U_i_g_simu.T@U_i_g_simu)@U_i_g.T@Y_i_qr
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

		# maximum between cluster deviance
		self.TSSFull = TSS_full_simu
		self.BSSFull = BSS_full_simu
		self.RSSFull = RSS_full_simu
		self.TSSReduced = TSS_reduced_simu
		self.BSSReduced = BSS_reduced_simu
		self.RSSReduced = RSS_reduced_simu
		self.PFFull = pseudoF_full
		self.PFReduced = pseudoF_reduced

		# convergence
		self.Enorm = 1/I*np.linalg.norm(X_i_jk_N - Z_i_qr@np.kron(C_k_r_simu, B_j_q_simu).T, 2)
		self.Fs = Fs_simu  # all objective functional values
		self.converged = converged_simu

		# classification of objects (labels)
		self.Labels = np.where(U_i_g_simu)[1]

		# ----------- End of result update for best loop --------------

		return self

