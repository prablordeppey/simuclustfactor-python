
import numpy as np
from simuclustfactor import tandem
from simuclustfactor.generate_dataset import GenerateDataset

# Data generation
seed = 106382
full_tensor_shape = (8,5,4)
reduced_tensor_shape = (3,3,2)
data = GenerateDataset(full_tensor_shape, reduced_tensor_shape, seed=seed).additive_noise()
X_i_jk = data.X_i_jk

# user-defined parameter inputs 
U_i_g = data.U_i_g
B_j_q = data.B_j_q
C_k_r = data.C_k_r

def test_TWCFTA():
	'''
	obtain the correct centroids from TWCFTA
	'''

	# verbose True, svd init
	cf = tandem.TWCFTA(verbose=True, seed=0).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(cf.Y_g_qr,2).round(3)
	assert Y_norm == 2.945, 'incorrect centroids for svd init, verbose True'

	# random init
	cf = tandem.TWCFTA(init='random', verbose=False, seed=0).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(cf.Y_g_qr,2).round(3)
	assert Y_norm == 2.995, 'incorrect centroids for random init'

	# svd init, inputs given
	cf = tandem.TWCFTA(verbose=False, seed=0, U_i_g=U_i_g, B_j_q=B_j_q, C_k_r=C_k_r).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(cf.Y_g_qr,2).round(3)
	assert Y_norm == 2.314, 'incorrect centroids for svd init'


def test_TWFCTA():
	'''
	obtain the correct centroids from TWFCTA
	'''

	# verbose True, svd init
	fc = tandem.TWFCTA(verbose=True, seed=0).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(fc.Y_g_qr,2).round(3)
	assert Y_norm == 2.42, 'incorrect centroids for svd init, verbose True'

	# random init
	fc = tandem.TWFCTA(init='random', verbose=False, seed=0).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(fc.Y_g_qr,2).round(3)
	assert Y_norm == 2.795, 'incorrect centroids for random init'

	# svd init, inputs given
	fc = tandem.TWFCTA(verbose=False, seed=0, U_i_g=U_i_g, B_j_q=B_j_q, C_k_r=C_k_r).fit(
		X_i_jk, full_tensor_shape, reduced_tensor_shape)
	Y_norm = np.linalg.norm(fc.Y_g_qr,2).round(3)
	assert Y_norm == 2.554, 'incorrect centroids for svd init'

