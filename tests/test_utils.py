
from simuclustfactor import utils


X = [[1,2,3,4,5],[5,1,9,5,7],[7,5,3,5,6]]
I,J,K = 5,5,3
G,Q,R = 2,3,1
random_state = 0


def test_OneKMeans():
	"""
	Performing 1 run of the KMeans algorithm.
	Get the membership allocation matrix
	"""
	U_i_g = utils.OneKMeans(X,G,seed=random_state)
	U_i_g_actual = [[1,0], [0,1], [0,1]]

	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent OneKMeans results'


def test_RandomMembershipMatrix():
	"""Generate random membership matrix
	"""
	U_i_g = utils.RandomMembershipMatrix(I,G, seed=random_state)
	U_i_g_actual = [[1,0],[0,1],[0,1],[0,1],[0,1]]
	assert (U_i_g == U_i_g_actual).all()==True, 'inconsistent random matrix'


def test_SingularVectors():
	'''
	return first entry of left singular vectors of X
	'''
	first_correct_entry = -0.37422995389883956
	res_from_implementation = utils.SingularVectors(X=X,D=1)[0][0]
	# print(utils.singular_vectors(X=X,D=1)[0][0])
	assert res_from_implementation == first_correct_entry, 'inaccurate left singular vectors'


def test_PseudoF():
	'''
	compute the pseudoF score weighted on the number of components and cluster
	'''

	bss = 48.23
	wss = 23.5
	pf_value = 14.366382978723403
	pf = utils.PseudoF(bss=bss, wss=wss, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
	
	assert pf == pf_value, 'inaccurate PseudoF index'
