#Contains matrix factorization methods
#Can be used for sign prediction in social networks
#Based on Chiang et. al, 2014
#and Koren, Bell, and Volinsky, 2009

import numpy as np
from scipy.sparse import csr_matrix, rand
from scipy.sparse.linalg import svds
from scipy.linalg import norm, lstsq

#TODO convert types (make things scipy sparse matrices)
#TODO debug, test

#Matrix factorization with stochastic gradient descent
#Input: matrix, 
#       learning rate/loss type/tolerance,max # of iters for SGD, 
#       regularization of objective function
#       dimensionality of matrix factor embeddings
#Output: two matrices that are approximately factors
#of original matrix
#As described in Koren, Bell, and Volinksy, 2009
def matrix_factor_SGD(matrix, 
                      learning_rate, 
                      loss_type,
                      tol, 
                      max_iters, 
                      regularization_param,
                      dim):
  #initialization of two factors: zeros
  factor1 = rand(matrix.shape[0], dim, density=1, format="csr")
  factor2 = rand(matrix.shape[0], dim, density=1, format="csr")

  num_iters = 0
  #iterate over all nonzero entries (training entries)
  #do unless stopping criterion is met 
  #(found good enough approximation or iterated long enough)
  while num_iters < max_iters and diff(matrix,factor1,factor2) > tol:
    num_iters += 1
    #TODO where is the stochastic part? maybe pick row,col at random
    rows, cols = matrix.nonzero()
    nonzero_entries = zip(rows, cols)
    for entry in nonzero_entries: 
      row = entry[0]
      col = entry[1]
      #compute loss
      actual_entry = matrix[row,col]
      approx_factor1 = np.asarray(factor1[row,:].A[0]) #shallow copy TODO is this necessary?
      approx_factor2 = np.asarray(factor2[col,:].A[0])
      approx_entry = np.dot(approx_factor1,approx_factor2)
      #loss on this training example
      tr_loss = loss(actual_entry,approx_entry,loss_type) 

      #update estimates
      factor1[row,:] = approx_factor1 + learning_rate * \
          (tr_loss*approx_factor2 - regularization_param * approx_factor1)
      factor2[col,:] = approx_factor2 + learning_rate * \
          (tr_loss*approx_factor1 - regularization_param * approx_factor2)

  print("Found estimate with %f error in %d iterations" % \
    (diff(matrix,factor1,factor2), num_iters))
  return factor1, factor2

#Compute tolerance criterion
#See if product of factors is close enough (by Frobenius norm)
#to original matrix
#Input: matrix, factors, tolerance
#Output: Frobenius norm of difference between matrix and product 

#TODO: in the spirit of this problem maybe calculate difference of signs?
def diff(matrix,factor1,factor2):
  difference = matrix - factor1*factor2.transpose()
  return norm(difference.A, "fro")

#Allow user to choose loss function
#Input: actual and predicted points to compute loss of
#  Loss function type
#Output value: loss from desired loss function
def loss(actual,pred,loss_type):
  if loss_type is "sigmoid":
    return sigmoid_loss(actual,pred)
  elif loss_type is "square-hinge":
    return square_hinge_loss(actual,pred)
  else:
    raise InputError("unrecognized type of loss")

#Compute sigmoid loss
#Input: actual and predicted points to compute loss of
#Output value: sigmoid loss
def sigmoid_loss(actual,pred):
  return 1/(1+np.exp(actual*pred))

#Compute square hinge loss
#Input: actual and predicted points to compute loss of
#Output value: square hinge loss
def square_hinge_loss(actual,pred):
  return (max(0,1-actual*pred))**2

#Matrix factorization with alternating least squares
#Input: matrix, dimensionality of matrix factor embeddings
#Output: two matrices that are approximately factors
#of original matrix
#TODO test, add stopping criterion based on convergence, not just max iters
def matrix_factor_ALS(matrix, dim, num_iters):
  #initialization of two factors: small random (between -1 and 1)
  #matrix.dtype = "float64" #TODO is this necessary
  #factor1 = rand(matrix.shape[0], dim, density=1, format="csr")
  #factor2 = rand(matrix.shape[0], dim, density=1, format="csr")
  factor1 = np.random.random((matrix.shape[0], dim))
  #factor2 = np.random.random((matrix.shape[0], dim))
  factor2 = np.random.random((dim, matrix.shape[0]))
  #mat = np.asarray(matrix.todense())
  #print mat
  #print type(mat)
  #print np.isnan(np.min(mat))
  for iteration in range(num_iters):
    #solve 2 least squares problems
    #one fixing the second factor and solving for the first
    #then fix first factor and solve for the second
    '''
    print factor1.shape, factor2.shape, matrix.shape
    print type(factor1), factor1.dtype
    print type(factor2), factor2.dtype
    print type(matrix), matrix.dtype
    a1,b1 = map(np.asarray, (factor2.A,matrix.A))
    #a1 = np.asarray(factor2.A)
    #b1 = np.asarray(matrix.A)
    print type(a1), type(b1)
    print a1.shape, b1.shape
    print len(a1.shape), len(b1.shape)
    '''
    factor1 = lstsq(factor2.transpose(), matrix.A)[0]
    #print factor1
    print "solved for factor 1"
    print factor1.shape, factor2.shape, matrix.shape
    factor2 = lstsq(factor1.transpose(), matrix.A)[0]
    print "solved for factor 2"
    '''
    print factor2.shape, mat.shape
    print mat[:10,:10]
    factor1 = np.linalg.lstsq(factor2[0], mat)
    print "solved for factor 1"
    factor2 = np.linalg.lstsq(factor1, mat)
    print "solved for factor 2"
    '''
  return csr_matrix(factor1), csr_matrix(factor2)


if __name__ == "__main__":
  pass

