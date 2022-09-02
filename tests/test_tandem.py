from simuclustfactor import tandem

X =[[1,2,3,4,5,3,2,4,3,7],
	[5,1,9,5,7,5,1,9,5,5],
	[7,5,3,5,6,3,0,7,5,3]]
I,J,K = 3,5,2
G,Q,R = 2,3,1


def test_TWCFTA():
	'''
	return first entry of left singular vectors of X
	'''
	Y_1_11 = -2.902108926131496
	cf = tandem.TWCFTA(init='svd', random_state=0).fit(X_i_jk=X, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
	assert cf.Y_g_qr[0][0] == Y_1_11, 'inaccurate left singular vectors'


def test_TWFCTA():
	
	fc = tandem.TWFCTA(init='svd', random_state=0).fit(X_i_jk=X, full_tensor_shape=(I,J,K), reduced_tensor_shape=(G,Q,R))
	Y_1_11 = -2.4432168410899413  # expectedfirst element of Y_g_qr matrix
	assert fc.Y_g_qr[0][0] == Y_1_11, 'inaccurate left singular vectors'

