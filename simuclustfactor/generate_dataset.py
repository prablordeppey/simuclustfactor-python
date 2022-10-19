
import numpy as np
from .utils import RandomMembershipMatrix

class GenerateDataset:

	def __init__(self, full_tensor_shape=(8,5,4), reduced_tensor_shape=(3,3,2), centroids_spread=[0,1], noise_mean=0, noise_stdev=0.5, seed=None, rng=None):
		"""
		The __init__ function initializes the parameters of the model.
		The default values are:
		I = 8, J = 5, K = 4, G = 3, Q=3 and R=2. 
		The noise mean is 0 and stdev is 0.5.
		
		:param tuple full_tensor_shape: (I,J,K) Used to Define the number of objects, variables and occassions respectively.
		:param tuple reduced_tensor_shape: (G,Q,R) Used to Define the number of clusters in the data, factors for varibles and occasions respectively.
		:param tuple centroids_spread: Used to Set the interval from which to uniformly pick centroids.
		:param float noise_mean=0: Used to Set the mean of the normal distribution used to generate random numbers.
		:param float noise_stdev=0.5: Used to Set the standard deviation of the normal distribution used to generate random numbers.
		:param int seed=None: Used to Set the seed of the random number generator.
		:param numpy.random._generator.Generator rng=None: Used to Set the random number generator.
		:return: The values of the parameters.
		
		:Example:
			:Example:
				>>> from generate_dataset import GenerateDataset
				>>> seed = 106382
				>>> full_tensor_shape = (8,5,4)
				>>> reduced_tensor_shape = (3,3,2)
				>>> data = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise()
				>>> X_i_jk = data.X_i_jk
				>>> Y_g_qr = data.Y_g_qr
				>>> U_i_g = data.U_i_g
				...
		"""
		self.I = full_tensor_shape[0]
		self.J = full_tensor_shape[1]
		self.K = full_tensor_shape[2]
		self.G = reduced_tensor_shape[0]
		self.Q = reduced_tensor_shape[1]
		self.R = reduced_tensor_shape[2]
		self.centroids_spread=centroids_spread
		self.noise_mean = noise_mean
		self.noise_stdev = noise_stdev
		self.seed = seed
		self.rng = np.random.default_rng(seed) if rng is None else rng  # defining random generator

	def additive_noise(self):
		"""
		The additive_noise function creates a random noise matrix E_i_jk, which is added to the latent factors Z_i_jk.
		The function also creates labels for each of the three membership matrices U, B and C.

		:param self: Used to Access variables that belong to the class.
		:return: The following: X_i_jk, Z_i_jk, E_i_jk, Y_g_qr, C_k_r, B_j_q, U_i_g, U_labels, B_labels, C_labels.

		:doc-author: Trelent
		"""

		# 8x3
		U_i_g = RandomMembershipMatrix(I=self.I, G=self.G, rng=self.rng)

		# 6x3
		B_j_q = RandomMembershipMatrix(I=self.J, G=self.Q, rng=self.rng)  # contruct membership matrix
		weights = 1/B_j_q.sum(axis=0)**0.5  # compute weigh ts for each factor
		facts = np.where(B_j_q==1)[1]  # get corresponding factors for each var
		B_j_q[B_j_q==1] = weights[facts]  # update weight of var-factor entry

		# 6x3
		C_k_r = RandomMembershipMatrix(I=self.K, G=self.R, rng=self.rng)  # contruct membership matrix
		weights = 1/C_k_r.sum(axis=0)**0.5  # compute weights for each factor
		facts = np.where(C_k_r==1)[1]  # get corresponding factors for each var
		C_k_r[C_k_r==1] = weights[facts]  # update weight of occ-factor entry

		Y_g_qr = self.rng.uniform(size=self.G*self.Q*self.R, low=self.centroids_spread[0], high=self.centroids_spread[1]).reshape(self.G,-1) # 3x6

		# 8x6
		Z_i_jk = U_i_g @ Y_g_qr @ np.kron(C_k_r, B_j_q).T

		# noise creation
		E_i_jk = self.rng.normal(loc=self.noise_mean, scale=self.noise_stdev, size=(self.I,self.J*self.K))

		# noise addition
		X_i_jk = Z_i_jk + E_i_jk

		# labels set
		self.U_labels = np.where(U_i_g)[1]
		self.B_labels = np.where(B_j_q)[1]
		self.C_labels = np.where(C_k_r)[1]
		
		# generated sets
		self.X_i_jk = X_i_jk
		self.Z_i_jk = Z_i_jk
		self.E_i_jk = E_i_jk
		self.Y_g_qr = Y_g_qr
		self.C_k_r = C_k_r
		self.B_j_q = B_j_q
		self.U_i_g = U_i_g

		return self
