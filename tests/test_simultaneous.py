
import numpy as np
from simuclustfactor import simultaneous
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

def test_T3Clus():
    '''
    obtain the correct centroids from T3Clus
    '''

    # verbose True, svd init
    t3 = simultaneous.T3Clus(verbose=True, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(t3.Y_g_qr,2).round(3)
    assert Y_norm == 4.131, 'incorrect centroids for svd init, verbose True'

    # random init
    t3 = simultaneous.T3Clus(init='random', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(t3.Y_g_qr,2).round(3)
    assert Y_norm == 3.756, 'incorrect centroids for random init'

    # twcfta init
    t3 = simultaneous.T3Clus(init='twcfta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(t3.Y_g_qr,2).round(3)
    assert Y_norm == 2.771, 'incorrect centroids for twcfta init'

    # twfcta init
    t3 = simultaneous.T3Clus(init='twfcta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(t3.Y_g_qr,2).round(3)
    assert Y_norm == 3.417, 'incorrect centroids for twcfta init'

    # svd init, inputs given
    t3 = simultaneous.T3Clus(verbose=False, seed=0, U_i_g=U_i_g, B_j_q=B_j_q, C_k_r=C_k_r).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(t3.Y_g_qr,2).round(3)
    print(Y_norm)
    assert Y_norm == 3.894, 'incorrect centroids for svd init'


def test_3FKMeans():
    '''
    obtain the correct centroids from 3FKMeans
    '''
    
    # verbose True, svd init
    tfk = simultaneous.TFKMeans(verbose=True, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(tfk.Y_g_qr,2).round(3)
    assert Y_norm == 3.149, 'incorrect centroids for svd init, verbose True'

    # random init
    tfk = simultaneous.TFKMeans(init='random', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(tfk.Y_g_qr,2).round(3)
    assert Y_norm == 2.901, 'incorrect centroids for random init'

    # twcfta init
    tfk = simultaneous.TFKMeans(init='twcfta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(tfk.Y_g_qr,2).round(3)
    assert Y_norm == 4.474, 'incorrect centroids for twcfta init'

    # twfcta init
    tfk = simultaneous.TFKMeans(init='twfcta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(tfk.Y_g_qr,2).round(3)
    assert Y_norm == 2.611, 'incorrect centroids for twcfta init'

    # svd init, inputs given
    tfk = simultaneous.TFKMeans(verbose=False, seed=0, U_i_g=U_i_g, B_j_q=B_j_q, C_k_r=C_k_r).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(tfk.Y_g_qr,2).round(3)
    print(Y_norm)
    assert Y_norm == 3.552, 'incorrect centroids for svd init'


def test_CT3Clus():
    '''
    obtain the correct centroids from CT3Clus
    '''

    # verbose True, svd init
    ct3 = simultaneous.CT3Clus(verbose=True, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(ct3.Y_g_qr,2).round(3)
    assert Y_norm == 3.454, 'incorrect centroids for svd init, verbose True'

    # random init
    ct3 = simultaneous.CT3Clus(init='random', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(ct3.Y_g_qr,2).round(3)
    assert Y_norm == 3.377, 'incorrect centroids for random init'

    # twcfta init
    ct3 = simultaneous.CT3Clus(init='twcfta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(ct3.Y_g_qr,2).round(3)
    assert Y_norm == 3.332, 'incorrect centroids for twcfta init'

    # twfcta init
    ct3 = simultaneous.CT3Clus(init='twfcta', verbose=False, seed=0).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(ct3.Y_g_qr,2).round(3)
    assert Y_norm == 3.085, 'incorrect centroids for twcfta init'

    # svd init, inputs given
    ct3 = simultaneous.CT3Clus(verbose=False, seed=0, U_i_g=U_i_g, B_j_q=B_j_q, C_k_r=C_k_r).fit(
        X_i_jk, full_tensor_shape, reduced_tensor_shape)
    Y_norm = np.linalg.norm(ct3.Y_g_qr,2).round(3)
    print(Y_norm)
    assert Y_norm == 2.506, 'incorrect centroids for svd init'
