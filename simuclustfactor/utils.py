### -------------------

# Helping functions to 

### -------------------

import numpy as np

# =========== 1 RUN OF THE KMEANS ALGORITHM

def OneKMeans(Y_i_qr, G, U_i_g=None, rng=None, seed=None):
	"""
	The OneKMeans function takes in the following parameters:
		- Y_i_qr: A matrix of the component scores.
		- G: The number of groups to cluster the objects into.
		- U_i_g=None (optional): A membership matrix for all objects, with each row representing an object and each column representing a group.

	:param Y_i_qr: Used to Store the data matrix for all objects.
	:param G: Used to Define the number of clusters.
	:param U_i_g=None: Used to Pass in a membership matrix to the function.
	:param rng=None: Used to Ensure that the results are reproducible.
	:param seed=None: Used to Ensure that the results are reproducible.
	:return: A stochastic membership matrix (U_i_g), allocating each component score to a centroid in the reduced space is returned.
	"""

	# defining random generator with no seed to radnom results.
	if rng is None:
		rng = np.random.default_rng(seed)

	Y_i_qr = np.array(Y_i_qr)

	I = Y_i_qr.shape[0]

	# initialize centroids matrix
	if U_i_g is None:
		U_i_g = RandomMembershipMatrix(I,G,rng=rng)
	else:
		U_i_g = np.array(U_i_g)
	Y_g_qr = np.diag(1/U_i_g.sum(axis=0)) @ U_i_g.T @ Y_i_qr

	# ------- case 1: repeat until no empty clustering	
	U_i_g = np.zeros((I,G))

	# assign each object to the respective cluster
	for i in range(I):
		dist = ((Y_i_qr[i,:]-Y_g_qr)**2).sum(axis=1)  # calculate distance between obj and centroids.
		min_dist_cluster = dist.argmin()  # get cluster with smallest distancee from object.
		U_i_g[i, min_dist_cluster] = 1  # assign the object to that cluster.

	# possibility of observing empty clusters
	C_g = U_i_g.sum(axis=0)  # get count of members in each cluster
	while (C_g==0).any():
		LC = C_g.argmax()  # select the largest cluster
		EC = np.where(C_g==0)[0][0]  # select next empty cluster
		
		LC_members = np.where(U_i_g[:,LC]==1)[0]

		LC_scores = Y_i_qr[LC_members,]
		U_i_g = _split_update(LC, LC_members, LC_scores, EC, U_i_g, C_g, seed)  # splitting cluster into 2 sub-clusters and updating U_i_g

		C_g = U_i_g.sum(axis=0)  # ensure the alg stops

	return U_i_g


def _split_update(LC, LC_members, LC_scores, EC, U_i_g, C_g, seed):
	"""
	The split_update function takes in the following parameters:
		- LC: The index of the largest cluster.
		- EC: The index of the empty cluster.
		- U_i_g: A membership matrix for all objects, with each row representing an object and each column representing a group.
	
		It then performs these operations to reassign objects to new clusters based on their distance from subcluster centroids (Y_2):
			- M = number of objects in largest cluster.
			- Find indices for all members of largest cluster.
			- Initialize a 2xQR centroids matrix Y_2 for subclusters using random initialization.
			- Allocate members to closest centroids.
	
		It returns the updated membership function matrix u_i_g.
	
	:param LC: Used to Select the largest cluster.
	:param LC_members: Used to Store the indices of all objects in the largest cluster.
	:param LC_scores: Used to Store the scores of each cluster.
	:param EC: Used to Store the index of the empty cluster.
	:param U_i_g: Used to Assign each object to a cluster.
	:param C_g: Used to Store the index of the largest cluster.
	:param seed: Used to Initialize the random number generator.
	:return: The updated membership matrix u_i_g.
	
	:doc-author: Trelent
	"""

	M = len(LC_members)  # number of objects in the largest cluster.

	# perform K-means to split members of LC into 2 clusters and update U_i_g matrix
	U_m_2 = RandomMembershipMatrix(I=M, G=2, seed=seed)  # initialize matrix with 2 groups
	Y_2_qr = np.diag(1/U_m_2.sum(axis=0)) @ U_m_2.T @ LC_scores   # 2xQR centroids matrix for subclusters

	# assign each cluster member to the respective sub-cluster
	for i in range(len(LC_scores)):

		dist = ((LC_scores[i,]-Y_2_qr)**2).sum(axis=1)  # calculate distance between obj and the 2 sub-centroids.
		min_dist_cluster = dist.argmin()  # get cluster with smallest distance.

		if (min_dist_cluster == 0):
			U_i_g[LC_members[i],] = 0  # unassign the obj from the large cluster.
			U_i_g[LC_members[i], EC] = 1  # assign the obj to the empty cluster.

	return U_i_g


# =========== BUILDING MEMBERSHIP FUNCTION MATRIX CONSTRUCTION

def RandomMembershipMatrix(I, G, rng=None, seed=None):
	"""
	The RandomMembershipMatrix function creates a random membership matrix U_i_g. 
	The function takes as input the number of objects I and the number of clusters G, 
	and returns a matrix U_i_g with dimensions (I x G) where each row is an object, and each column is a cluster. 
	The first G rows are assigned to unique clusters 0 through G-2. The last I-G rows are randomly assigned to one of the existing clusters.
	
	:param I: Used to define the number of objects in the dataset.
	:param G: Used to define the number of clusters.
	:param rng=None: Used to specify the random number generator.
	:param seed=None: Used to ensure that the results are random.
	:return: A binary stochastic matrix allocating objects to a cluster.
	"""
	
	# defining random generator with no seed to radnom results.
	if rng is None:
		rng = np.random.default_rng(seed)

	# initialize U_i_g
	U_i_g = np.zeros((I,G))
	U_i_g[:G,] = np.eye(G)  # first G assignments to unique clusters. To ensure no cluster is empty

	# assign random clusters to remaining objects
	if I > G:
		for p in range(G,I):
			c = rng.integers(G) # choose a random cluster for the i'th object.
			U_i_g[p,c] = 1  # assign object p to cluster c

	return U_i_g

# ===========  END OF MEMBERSHIP FUNCTION CONSTRUCTION


# ===========  LARGEST EIGENVECTORS

def EigenVectors(X, D):
	"""
	The EigenVectors function takes in a matrix X and the number of desired eigenvectors D.
	It returns the first D eigenvectors of X, sorted by their corresponding eigenvalues.
	eigh instead of eig because of truncation errors with numpy's implementtation which could 
	result in incorrect result (eg. complex vectors for covariance matrices)


	:param X: Used to Calculate the eigenvectors of x.
	:param D: Used to Specify the number of eigenvectors to return.
	:return: The eigenvectors of the covariance matrix.
	
	"""
	
	# U,_,_ = np.linalg.svd(X)
	# return U[:,:D]

	eigenValues,eigVectors = np.linalg.eigh(X)
	# eigenValues,eigVectors = np.linalg.eig(X)
	idx = eigenValues.argsort()[::-1]  
	eigenValues = eigenValues[idx]

	return eigVectors[:,:D]

# ===========  END OF LARGEST EIGENVECTORS


# ===========  START OF THE PseudoF STATISTIC IN FULL & REDUCED spaces

def PseudoF_Full(bss, wss, full_tensor_shape, reduced_tensor_shape):
	"""
	The PseudoF function is a measure of the ratio between the amount of 
	variance explained by a model with `rank` components, and the variance 
	explained by an optimal model with `tensor_shape` dimensions. This function 
	is used to select an appropriate rank for a tensor decomposition.
	
	:param bss: Between cluster sum of squares deviance.
	:type bss: float
	:param wss: Within cluster sum of squares deviance.
	:type wss: float
	:param full_tensor_shape: (I,J,K) tensor dimensions.
	:type full_tensor_shape: tuple
	:param reduced_tensor_shape: (G,Q,R) G clusters, Q components for variables, R components for Occasions.
	:type reduced_tensor_shape: tuple
	:return: The pseudo-f statistic for the given tensor and rank.
	:rtype: float
	
	Reference:
		[1] Roberto Rocci and Maurizio Vichi (2005).
		Three-mode component analysis with crisp or fuzzy partition of units. 
		Psychometrika, 70:715–736, 02 2005.
	"""
	I,J,K = full_tensor_shape
	G,Q,R = reduced_tensor_shape
	db = (G-1)*Q*R + (J-Q)*Q + (K-R)*R
	dw = I*J*K - (G*Q*R + (J-Q)*Q + (K-R)*R)
	return (bss/db)/(wss/dw)


def PseudoF_Reduced(bss, wss, full_tensor_shape, reduced_tensor_shape):
	"""
	The PseudoF function is a measure of the ratio between the amount of 
	variance explained by a model with `rank` components, and the variance 
	explained by an optimal model with `tensor_shape` dimensions. Computes
	the PseudoF score without taking into account any contraints. 
	The naive implementation.
	
	:param bss: Between cluster sum of squares deviance.
	:type bss: float
	:param wss: Within cluster sum of squares deviance.
	:type wss: float
	:param full_tensor_shape: (I,J,K) tensor dimensions.
	:type full_tensor_shape: tuple
	:param reduced_tensor_shape: (G,Q,R) G clusters, Q components for variables, R components for Occasions.
	:type reduced_tensor_shape: tuple
	:return: The pseudo-f statistic for the given tensor and rank.
	:rtype: float
	
	Reference:
		[1] T. Caliński & J Harabasz (1974).
		A dendrite method for cluster analysis
		Communications in Statistics, 3:1, 1-27, DOI: 10.1080/03610927408827101
	"""
	I,J,K = full_tensor_shape
	G,Q,R = reduced_tensor_shape
	db = G-1
	dw = I*Q*R - G
	return (bss/db)/(wss/dw)

# ===========  END OF PseudoF STATISTIC FUNCTION


# ===========  START OF _BASECLASS

class _BaseClass:
	"""
	Base class for the tandem tucker-factorial and kmeans-clustering models.
	For checking initialization configuration.
	
	:param bool verbose: whether to display executions output or not. Defaults to False.
	:param str init: the parameter initialization method. Defaults to svd.
	:param int seed: seed for random sequence generation. Defaults to None.
	:param int n_max_iter: maximum number of iterations. Defaults to 10.
	:param int n_loops: number of random initializations to gurantee global results. Defaults to 10.
	:param float tol: tolerance level/acceptable error. Defaults to 1e-5.
	"""

	def __init__(
		self,
		seed=None,
		verbose=False,
		init='svd',
		n_max_iter=10,
		n_loops=10,
		tol=1e-3,
		U_i_g=None,
		B_j_q=None,
		C_k_r=None
	):
		self.init = init
		self.n_max_iter = n_max_iter
		self.n_loops = n_loops
		self.tol = tol
		self.seed = seed
		self.verbose = verbose

		self.full_tensor_shape = None  # (I,J,K)
		self.reduced_tensor_shape = None  # (G,Q,R)
		self.B_j_q = B_j_q
		self.C_k_r = C_k_r
		self.U_i_g = U_i_g

	# Check initialization configurations.
	def _check_params(self):
		"""
		verify valid input tensor dimensions in the full and reduced space.
		"""
		# n_max_iter
		if (not isinstance(self.n_max_iter, int)) or (not self.n_max_iter > 0):
			raise ValueError(f"n_max_iter should be > 0, got {self.n_max_iter} instead.")

		# tol
		if (not isinstance(self.tol, float)) or (not 0 < self.tol < 1):
			raise ValueError(
				f"tolerance should be very small positive number between < 1 but got {self.tol}"
			)

		# verbose
		if not isinstance(self.verbose, bool):
			raise ValueError(
				f"verbose must be boolean but got {type(self.verbose).__name__}"
			)

		# ensure all dimensions given are greater than 0
		if not (np.array(self.reduced_tensor_shape) <= np.array(self.full_tensor_shape)).all():
			raise ValueError(
				f"reduced_tensor_shape={self.reduced_tensor_shape} must be <= full_tensor_shape={self.full_tensor_shape}."
			)

	# component matrices validation
	def _check_initialized_components(self):
		"""
		If U_i_g,B_j_q and C_k_r are user-defined, dimensions of these matrices are validated.
		U_i_g must also be row-stochastic (row-sums equal to 1)
		"""
		# check U membership matrix
		if self.U_i_g is not None:
			# check dimension
			if self.U_i_g.shape != (self.full_tensor_shape[0], self.reduced_tensor_shape[0]):
				raise ValueError(
					f"incorrect U_i_g matrix, expected shape {(self.full_tensor_shape[0], self.reduced_tensor_shape[0])} but got {self.U_i_g.shape}"
				)
			
			# check row stochastic nature
			if (self.U_i_g.sum(axis=0) != 0).all():
				raise ValueError(
					f"incorrect U_i_g matrix. U_i_g must be row stochastic."
				)

		# check B component matrix
		if self.B_j_q is not None:
			# check dimension
			if self.B_j_q.shape != (self.full_tensor_shape[1], self.reduced_tensor_shape[1]):
				raise ValueError(
					f"incorrect B_j_q matrix, expected shape {(self.full_tensor_shape[1], self.reduced_tensor_shape[1])} but got {self.B_j_q.shape}"
				)
		
		# check C component matrix
		if self.C_k_r is not None:
			# check dimension
			if self.C_k_r.shape != (self.full_tensor_shape[2], self.reduced_tensor_shape[2]):
				raise ValueError(
					f"incorrect C_k_r matrix, expected shape {(self.full_tensor_shape[2], self.reduced_tensor_shape[2])} but got {self.C_k_r.shape}"
				)



