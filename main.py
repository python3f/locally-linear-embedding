from sklearn.neighbors import kneighbors_graph
from scipy.sparse      import csr_matrix
from numpy             import matlib
import numpy as np


def csr_from_mat(W, NI):
  n, k = W.shape
  data = np.reshape(W, n*k)
  cols = np.reshape(NI, n*k)
  rows = np.floor(np.arange(0, n, 1/k))
  return csr_matrix((data, (rows, cols)), shape=(n, n))

def lle_neighborhood(X, k):
  n, d = X.shape
  NN = kneighbors_graph(X, k, mode='connectivity')
  return np.reshape(NN.indices, (n, k))

def lle_weights(X, NI):
  n, d = X.shape
  n, k = NI.shape
  tol  = 1e-3 if k>d else 0
  W = np.zeros((n, k))
  for i in range(n):
    Z = (X[NI[i,:],:] - matlib.repmat(X[i,:], k, 1)).T
    C = Z.T.dot(Z)
    C = C + tol*np.trace(C)*np.identity(k)
    w = np.linalg.inv(C).dot(np.ones((k, 1)))
    w = w / np.sum(w)
    W[i,:] = w.T
  return W

def lle_embedding(W, m):
  n, n = W.shape
  I, W = np.identity(n), W
  M = (I-W).T.dot(I-W)
  w, v = np.linalg.eig(M)
  i = np.argsort(w)
  w, v = w[i].real, v[:,i].real
  # did i do wrong here?
  return v[:,1:m+1]

"""Args:
X: input samples, array (num, dim)
n_components: dimension of output data
n_neighbours: neighborhood size

Returns:
Y: output samples, array (num, n_components)
"""
def LLE(X, n_components=2, n_neighbours=10):
  NI = lle_neighborhood(X, n_neighbours)
  W  = lle_weights(X, NI)
  W  = csr_from_mat(W, NI)
  Y  = lle_embedding(W, n_components)
  return Y
