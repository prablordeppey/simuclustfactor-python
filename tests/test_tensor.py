
from simuclustfactor import tensor


X_i_j_k = [[[1,2,3],[0,3,6]],[[5,1,9],[9,1,4]],[[7,5,6],[3,6,7]]]
I,J,K = 2,3,3
G,Q,R = 2,3,1
random_state = 0

def test_Unfold():
	"""
	Performing unfolding/matricization of tensors to matrix
	"""
	# mode K matricization
	X_k_ij = tensor.Unfold(X_i_j_k, mode=0)
	X_k_ij0_actual = [1,2,3,0,3,6]
	# print(X_k_ij[0])

	# mode I matricization
	X_i_jk = tensor.Unfold(X_i_j_k, mode=1)
	X_i_jk0_actual = [1,2,3,5,1,9,7,5,6]
	# print(X_i_jk[0])	

	# mode J matricization
	X_j_ki = tensor.Unfold(X_i_j_k, mode=2)
	X_j_ki0_actual = [1,5,7,0,9,3]
	# print(X_j_ki[0])

	assert ((X_k_ij[0] == X_k_ij0_actual).all() == True) and \
		((X_i_jk[0] == X_i_jk0_actual).all() == True) and \
		((X_j_ki[0] == X_j_ki0_actual).all() == True)	, 'inconsistent unfolding'

import numpy as np
def test_Fold():
	"""
	Folding matrix back to tensor of given dimension and rank.
	"""
	# mode-K folding
	X_k_ij = tensor.Unfold(X_i_j_k, mode=0)
	X_i_j_k0 = tensor.Fold(X_k_ij, mode=0, shape=(K,I,J))

	# mode-I folding
	X_i_jk = tensor.Unfold(X_i_j_k, mode=1)
	X_i_j_k1 = tensor.Fold(X_i_jk, mode=1, shape=(K,I,J))

	# mode-I folding
	X_j_ki = tensor.Unfold(X_i_j_k, mode=2)
	X_i_j_k2 = tensor.Fold(X_j_ki, mode=2, shape=(K,I,J))

	assert ((X_i_j_k0 == X_i_j_k).all() == True) and \
		((X_i_j_k1 == X_i_j_k).all() == True) and \
		((X_i_j_k2 == X_i_j_k).all() == True)	, 'inconsistent folding'
