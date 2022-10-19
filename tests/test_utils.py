
import numpy as np
from simuclustfactor import utils
from simuclustfactor.generate_dataset import GenerateDataset

# Data generation
seed = 106382
I,J,K = 8,5,4
G,Q,R = 3,3,2
full_tensor_shape = (I,J,K)
reduced_tensor_shape = (G,Q,R)
data = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise()
X_i_jk = data.X_i_jk

def test_PseudoF_Full():
	'''
	compute the pseudoF score in the full space.
	'''
	bss = 45
	wss = 20
	pf = utils.PseudoF_Full(bss=bss, wss=wss, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
	assert pf == 13.5, 'incorrect PseudoF_Full score'


def test_PseudoF_Reduced():
	'''
	compute the pseudoF score in the reduced space.
	'''
	bss = 45
	wss = 20
	pf = utils.PseudoF_Reduced(bss=bss, wss=wss, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
	assert pf == 50.625, 'incorrect PseudoF_Reduced score'


def test_OneKMeans():
	"""
	Performing one-run of the K-means algorithm.
	Get the membership allocation matrix
	"""
	# no U_i_g to update
	U_i_g = utils.OneKMeans(X_i_jk, G, seed=seed)
	U_i_g_actual = [[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent OneKMeans results'

	# updating U_i_g
	U_i_g0 = [[1, 0, 0],[0, 0, 0],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	U_i_g = utils.OneKMeans(X_i_jk, G, U_i_g=U_i_g0)
	U_i_g_actual = [[1, 0, 0],[1, 0, 0],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent OneKMeans results'

	# trigger empty clustering U_i_g
	U_i_g0 = [[1, 1, 0],[0, 0, 0],[0, 0, 1],[0, 0, 1],[0, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	U_i_g = utils.OneKMeans(X_i_jk, G, U_i_g=U_i_g0, seed=seed)
	U_i_g_actual = [[0, 1, 0],[1, 0, 0],[0, 0, 1],[0, 0, 1],[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	print(U_i_g)
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent OneKMeans results'



def test_SplitUpdate():

	# no U_i_g to update
	U_i_g = np.array([[1, 0, 0],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1],[1, 0, 0],[1, 0, 0],[1, 0, 0]])
	LC, EC = 2,1
	LC_members = [1,2,3,4]
	LC_scores = X_i_jk[:4,]
	C_g = U_i_g.sum(axis=0)
	U_i_g = utils._split_update(LC, LC_members, LC_scores, EC, U_i_g, C_g, seed)
	
	U_i_g_actual = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 0, 1],[0, 0, 1],[1, 0, 0],[1, 0, 0],[1, 0, 0]])
	
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent OneKMeans results'


def test_RandomMembershipMatrix():
	"""Generate random membership matrix
	"""
	U_i_g = utils.RandomMembershipMatrix(I,G, seed=seed)
	U_i_g_actual = [[1, 0, 0],[0, 1, 0],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0]]
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent random matrix'


def test_EigenVectors():
	'''
	return first entry of left singular vectors of X
	'''
	f_entry = 0.870
	entry = utils.EigenVectors(X_i_jk.T@X_i_jk, 2)[0,0].round(3)
	assert entry == f_entry, 'inaccurate eigenvectors'

def test_BaseClass():
	import pytest

	base = utils._BaseClass()  # initialize object

	assert type(base).__name__ == '_BaseClass', '_BaseClass name changed'

	# n_max_iter check
	base.n_max_iter = '10'
	try:
		base._check_params()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect n_max_iter definition'

	# tol check
	base.n_max_iter = 10
	base.tol = True
	try:
		base._check_params()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tol definition'

	# verbose check
	base.n_max_iter = 10
	base.tol = 1e-5
	base.verbose = 34
	try:
		base._check_params()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect verbose flag definition'

	# tensor_shapes check
	base.n_max_iter = 10
	base.tol = 1e-5
	base.verbose = True
	base.full_tensor_shape = (3,3,2)
	base.reduced_tensor_shape = (8,5,4)
	try:
		base._check_params()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tensor shapes definition'


	# checking initialized params
	base = utils._BaseClass()  # initialize object 

	# check U_i_g inconsistent dim
	base.full_tensor_shape = full_tensor_shape
	base.reduced_tensor_shape = reduced_tensor_shape
	base.U_i_g = utils.RandomMembershipMatrix(I+1, G)
	try:
		base._check_initialized_components()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tensor shapes definition'


	# check if a cluster empty
	U_i_g = utils.RandomMembershipMatrix(I, G)
	U_i_g[:,0] = 0
	base.U_i_g = U_i_g
	try:
		base._check_initialized_components()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tensor shapes definition'


	# check B_j_q inconsistent dim
	base.full_tensor_shape = full_tensor_shape
	base.reduced_tensor_shape = reduced_tensor_shape
	base.U_i_g = None
	base.B_j_q = utils.RandomMembershipMatrix(J, Q+1)
	try:
		base._check_initialized_components()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tensor shapes definition'


	# check C_k_r inconsistent dim
	base.full_tensor_shape = full_tensor_shape
	base.reduced_tensor_shape = reduced_tensor_shape
	base.U_i_g = None
	base.B_j_q = None
	base.C_k_r = utils.RandomMembershipMatrix(K, R+1)
	try:
		base._check_initialized_components()
	except Exception as e:
		assert type(e).__name__=='ValueError', 'incorrect tensor shapes definition'




