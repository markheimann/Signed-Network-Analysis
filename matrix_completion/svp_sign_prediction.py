#Contains methods for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.linalg import norm
import logging

#Input: adjacency matrix with signs as entries
# Rank (max rank of solution before taking sign matrix)
# Tolerance (how close does solution need to be to stop iterating)
# Maximum number of iterations
# Step size: how fast to update solution
# Mode: normal or test mode to perform additional tests
#       (only called by unit tests)

#Output: signed version low rank matrix that approximately solves
# sign prediction optimization problem
#NOTE: signed version is probably not low rank
#TODO write more official tests
def sign_prediction_SVP(adj_matrix, 
                        rank, 
                        tol, 
                        max_iter,
                        step_size,
                        mode = "normal"):

  #Initialization
  num_iters = 1 #start counting from 1
  dens = 0.01 #initialize factors with this density
  solution = 2*sp.rand(adj_matrix.shape[0], adj_matrix.shape[1], density=dens, format="csr")

  #Iterate until tolerance level or maximum # iters is reached
  while num_iters <= max_iter and not within_tol(solution, adj_matrix, tol):
    #update
    print "projection"
    rows, cols = adj_matrix.nonzero()
    solution = solution - step_size*(projection(
                          solution, rows, cols) - adj_matrix)
    #compute top <rank> SVs
    solution = solution.asfptype()
    print "svd"
    left_svecs, svals, right_svecs = svd(solution.A)
    
    #form low rank approximation
    solution = sp.csr_matrix(np.dot(np.dot(left_svecs[:,:rank], np.diag(svals[:rank])), right_svecs[:rank,:]))
    print "completed iteration ", num_iters
    num_iters += 1

  #confirm that solution (before signing, which will change things) is desired rank
  if mode == "test":
    assert np.linalg.matrix_rank(np.asarray(solution.todense())) == rank

  return sp.csr_matrix.sign(solution) #recall we want signs (edge sign predictions)

#Input: "solution" of SVP, adjacency matrix, tolerance
#Output: boolean reflecting whether or not
# projection of solution (onto nonzero elements of adj matrix)
# is "tolerably" close to adjacency matrix
def within_tol(solution, adj_matrix, tol):
  rows, cols = adj_matrix.nonzero()
  proj = projection(solution, rows, cols)
  diff = proj - adj_matrix
  try:
    return norm(diff.A, "fro") < tol #.A gets nonzero entries of sparse matrix
  except ValueError:
    logging.exception("Exception: ")
    print("%d NaNs in solution" % np.any(np.isnan(solution.A)))

#Input: matrix to project
#       rows, columns that can be nonzero in projection
#Output: projection (keep observed indices)
def projection(matrix, rows, cols):
  proj = sp.csr_matrix((matrix[rows,cols].A[0], (rows,cols)), shape=matrix.shape)
  return proj
