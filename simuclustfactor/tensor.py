import numpy as np


def Unfold(tensor, mode):
	"""
	The Unfold function takes a tensor and returns the unfolded version of it.
	The mode parameter specifies which dimension to unfold.
	For example, if the input is a matrix X, then Unfold(X,0) returns X_k_ij as a vector, Unfold(X,1) returns X_i_jk and Unfold(X,2) returns X_j_ki.
	
	:param tensor: The tensor to unfold.
	:type tensor: ndarray
	:param mode: Used to Specify which mode of the tensor is to be unfolded.
	:type mode: integer
	:return: A matrix of size (k,ij) for mode=0, (i,jk) for mode=2 and (j,ki) for mode=2.
	:rtype: ndarray

	Example:
		# mode 0 unfolding
		>> Unfold(X_i_j_k, mode=0)
		>> X_i_jk

		# mode 1 unfolding
		>> Unfold(X_i_j_k, mode=1)
		>> X_i_jk

		# mode 2 unfolding
		>> Unfold(X_i_j_k, mode=2)
		>> X_j_ki
	"""
	
	tensor = np.array(tensor)

	# chceks
	if not isinstance(mode, int): raise ValueError(f'mode is expected to be a number but got {{type(mode).__name__}}')
	if mode not in range(len(tensor.shape)): raise ValueError(f'possible modes are {range(len(tensor.shape))}. refer to the manual for appropriate valid mode.')  # only valid specified mode
	
	# main
	if (mode==0):  # (K,IJ)
		unfolded = np.array([X.flatten() for X in tensor])
	if (mode==1):  # (I,JK)
		unfolded = np.concatenate(tensor, axis=1)
	if (mode==2):  # (J,KI)
		unfolded = []
		for i in range(tensor.shape[1]):
			face = []
			for k in range(tensor.shape[0]):
				face.append(tensor[k][i])
			face = np.array(face).T
			unfolded.append(face)
		unfolded = np.hstack(unfolded)

	return unfolded


def Fold(X, mode, shape):
	"""
	Folding a matrix X back into a tensor X_i_j_k.
	The Fold function takes the following arguments: 
	mode: The mode to be folded. Must be an integer between 0 and 2 inclusive. 
	shape: The original shape of the tensor X (K,I,J).

	:param X: Used to Pass the tensor to be folded.
	:type X: ndarray
	:param mode: Used to Specify the way in which we want to fold the tensor.
	:type mode: int
	:param shape: Used to Specify the shape of the tensor.
	:type shape: tuple 
	:return: The folded original tensor shape.
	:rtype: ndarray

	Example:
		# mode 0 folding
		>> Fold(X_k_ij, mode=0, shape=(K,I,J))
		>> X_i_j_k

		# mode 1 folding
		>> Fold(X_i_jk, mode=1, shape=(K,I,J))
		>> X_i_j_k

		# mode 2 folding
		>> Fold(X_j_ki, mode=2, shape=(K,I,J))
		>> X_i_j_k
	"""

	X = np.array(X)

	# checks
	if not isinstance(mode, int): raise ValueError(f'mode expected to be a number but got {type(mode).__name__()}')
	if mode not in range(len(shape)): raise ValueError(f'mode must be a valid tensor mode, but got mode={mode}. check manual for correct tensor modes')
	if not len(X.shape)==2: raise ValueError(f'X must be a matrix of size m*n, but got { X.shape }')
	if not len(shape)==3: raise ValueError(f'shape must be a three-way tensor shape, but got shape={shape}')
	if X.size!=np.prod(shape): raise ValueError(f'shape size {np.prod(shape)} is not consistent with tensor size {X.size}')

	# main
	if (mode==0): # (K,IJ) => (K,I,J)
		folded = np.array([x.reshape(shape[1],-1) for x in X])

	if (mode==1): # (I,JK) => (K,I,J)
		folded = np.array([X[:,shape[-1]*ind:shape[-1]*(ind+1)] for ind in range(X.shape[1]//shape[-1])])

	if (mode==2): # (J,KI) => (K,I,J)
		# folded = np.array([X[:,shape[1]*ind:shape[1]*(ind+1)].T for ind in range(X.shape[1]//shape[1])])
		folded = np.array([X.T[k::shape[0]] for k in range(shape[0])])

	return folded


class Mat2Tensor(np.ndarray):
	"""
	It's role is to instantiate (create) the object, and return it. 
	The Mat2Tensor class is used to restructure a given matrix with faces stacked row/column wise.
	
	:param input_array: Used to Pass the array that is to be converted.
	:param I: Number of objects/units of the dataset
	:type I: integer
	:param J: Number of variables in the dataset.
	:type J: integer
	:param stacked='row': Used to Specify whether the faces are row-stacked ('row') or column-stacked ('coloumn').
	:type stacked: string
	:return: Reformatted matrix into correct matricized representation.
	:rtype: ndarray
	"""

	def __new__(cls, input_array, I, J, stacked='row'):
		
		obj = np.asarray(input_array)

		# checks
		if obj is None: raise ValueError('input_array cannot be None')
		if len(obj.shape)!=2: raise ValueError(f'input_array must be a matrix, but got shape {obj.shape}')

		# main
		if stacked=='column':
			obj = np.array([obj[:,J*ind:J*(ind+1)] for ind in range(obj.shape[1]//J)])  # reorder column stacked faces
		else:  # row
			obj = np.array([obj[I*ind:I*(ind+1),:] for ind in range(obj.shape[0]//I)])  # reorder rowstacked faces

		obj = np.asarray(obj).view(cls)
		return obj

