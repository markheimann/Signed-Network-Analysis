#Contains methods for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix, rand
from scipy.sparse.linalg import svds
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
  dens = 0.01#math.log(adj_matrix.shape[0])/adj_matrix.shape[0] #initialize factors with this density
  solution = 2*rand(adj_matrix.shape[0], adj_matrix.shape[1], density=dens, format="csr")
  #solution = csr_matrix(np.zeros(adj_matrix.shape)) #matrix of zeros

  #Iterate until tolerance level or maximum # iters is reached
  while num_iters <= max_iter and not within_tol(solution, adj_matrix, tol):
    #update
    print "projection"
    rows, cols = adj_matrix.nonzero()
    solution = solution - step_size*(projection(
                          solution, rows, cols) - adj_matrix)
    #compute top <rank> SVs
    #print solution.A
    solution = solution.asfptype()
    #left_svecs, svals, right_svecs = svds(solution.A, k = rank)
    print "svd"
    left_svecs, svals, right_svecs = svd(solution.A)
    '''
    print "Left svecs:", left_svecs.shape
    print left_svecs
    print
    print "svals:"
    print svals 
    print
    print "Right svecs:", right_svecs.shape
    print right_svecs
    print
    '''
    
    #form low rank approximation
    #solution = csr_matrix(np.dot(np.dot(left_svecs, np.diag(svals)), right_svecs))
    solution = csr_matrix(np.dot(np.dot(left_svecs[:,:rank], np.diag(svals[:rank])), right_svecs[:rank,:]))
    print "completed iteration ", num_iters
    num_iters += 1

  #confirm that solution (before signing, which will change things) is desired rank
  if mode == "test":
    assert np.linalg.matrix_rank(np.asarray(solution.todense())) == rank

  return csr_matrix.sign(solution) #recall we want signs (edge sign predictions)

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
  #proj = csr_matrix(np.zeros(matrix.shape)) #matrix of zeros
  proj = csr_matrix((matrix[rows,cols].A[0], (rows,cols)), shape=matrix.shape)

  #fill in with projected values
  #proj[observed_indices] = matrix[observed_indices] 
  return proj
