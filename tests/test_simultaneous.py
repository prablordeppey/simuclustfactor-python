
from simuclustfactor import simultaneous

X =[[1,2,3,4,5,3,2,4,3,7],
	[5,1,9,5,7,5,1,9,5,5],
	[7,5,3,5,6,3,0,7,5,3]]
I,J,K = 3,5,2
G,Q,R = 2,3,1

def test_T3Clus():
    '''
    make sure a is returned
    '''
    Y_1_11 = -2.5931220100429426
    ct3 = simultaneous.T3Clus(init='random', verbose=False, random_state=0).fit(X_i_jk=X, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
    # print(ct3.Y_g_qr[0][0])
    assert ct3.Y_g_qr[0][0] == Y_1_11, 'inaccurate left singular vectors'


def test_3FKMeans():
    '''
    make sure a is returned
    '''
    Y_1_11 = -0.975789907937135
    tfk = simultaneous.TFKMeans(init='random', verbose=False, random_state=0).fit(X_i_jk=X, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
    # print(tfk.Y_g_qr[0][0])
    assert tfk.Y_g_qr[0][0] == Y_1_11, 'inaccurate left singular vectors'


def test_CT3Clus():
    '''
    CT3Clus implementation
    '''
    Y_1_11 = -0.975789907937135
    alpha = 0  # 3FKMeans
    t3c0 = simultaneous.CT3Clus(init='random', verbose=False, random_state=0).fit(X_i_jk=X, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R), alpha=alpha)
    # print(t3c0.Y_g_qr[0][0])
    assert t3c0.Y_g_qr[0][0] == Y_1_11, 'inaccurate left singular vectors'

