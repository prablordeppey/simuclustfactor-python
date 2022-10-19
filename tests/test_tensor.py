import numpy as np
from simuclustfactor import tensor
from simuclustfactor.generate_dataset import GenerateDataset

# Data generation
seed = 106382
I,J,K = 8,5,4
G,Q,R = 3,3,2
full_tensor_shape = (I,J,K)
reduced_tensor_shape = (G,Q,R)
data = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise()
X_i_jk = data.X_i_jk

X_i_j_k = tensor.Fold(X_i_jk, 1, shape=(K,I,J))

# user-defined parameter inputs 
U_i_g = data.U_i_g
B_j_q = data.B_j_q
C_k_r = data.C_k_r

def test_Unfold():
	"""
	Performing unfolding/matricization of tensors to matrix
	"""
	# mode K matricization
	X_k_ij = tensor.Unfold(X_i_j_k, mode=0)
	assert np.linalg.norm(X_k_ij,2).round(3) == 4.823, "inconsistent mode-0 unfolding"

	# mode I matricization
	X_i_jk = tensor.Unfold(X_i_j_k, mode=1)
	assert np.linalg.norm(X_i_jk,2).round(3) == 4.314, "inconsistent mode-1 unfolding"

	# mode J matricization
	X_j_ki = tensor.Unfold(X_i_j_k, mode=2)
	assert np.linalg.norm(X_j_ki,2).round(3) == 4.785, "inconsistent mode-2 unfolding"


def test_Fold():
	"""
	Folding matrix back to tensor of given dimension and rank.
	"""
	# mode-K folding
	X_k_ij = tensor.Unfold(X_i_j_k, mode=0)
	X_i_j_k0 = tensor.Fold(X_k_ij, mode=0, shape=(K,I,J))
	assert np.linalg.norm(X_i_j_k0[0],2).round(3) == np.linalg.norm(X_i_j_k[0],2).round(3), "inconsistent mode-0 folding"

	# mode-I folding
	X_i_jk = tensor.Unfold(X_i_j_k, mode=1)
	X_i_j_k1 = tensor.Fold(X_i_jk, mode=1, shape=(K,I,J))
	assert np.linalg.norm(X_i_j_k1[0],2).round(3) == np.linalg.norm(X_i_j_k[0],2).round(3), "inconsistent mode-1 folding"

	# mode-I folding
	X_j_ki = tensor.Unfold(X_i_j_k, mode=2)
	X_i_j_k2 = tensor.Fold(X_j_ki, mode=2, shape=(K,I,J))
	assert np.linalg.norm(X_i_j_k2[0],2).round(3) == np.linalg.norm(X_i_j_k[0],2).round(3), "inconsistent mode-2 folding"

	