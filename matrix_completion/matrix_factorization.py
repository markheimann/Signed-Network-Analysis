#Contains matrix factorization methods
#Can be used for sign prediction in social networks
#Based on Chiang et. al, 2014
#and Koren, Bell, and Volinsky, 2009

import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm, lstsq
import math

#NOTE: SGD doesn't work very well--???

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
  matrix_density = matrix.nnz/(matrix.shape[0] * matrix.shape[1])
  #initialization of two factors: small random numbers between -0.01 and 0.01
  #may not want to initialize completely dense matrix or will take a while
  dens = 1 #initialize factors with this density
  factor1 = -4*sp.rand(matrix.shape[0], dim, density=dens, format="csr")
  fac1_rows, fac1_cols = factor1.nonzero()
  factor2 = -4*sp.rand(matrix.shape[0], dim, density=dens, format="csr")
  fac2_rows, fac2_cols = factor2.nonzero()

  num_iters = 1 #start counting from 1
  #iterate over all nonzero entries (training entries)
  #do unless stopping criterion is met 
  #(found good enough approximation or iterated long enough)

  error = np.inf
  errors = list()
  while num_iters <= max_iters and error > tol:
    rows, cols = matrix.nonzero()
    entry = np.random.randint(len(rows)) #choose entry at random
    row = rows[entry]
    col = cols[entry]

    #compute loss
    actual_entry = matrix[row,col]
    approx_factor1 = np.asarray(factor1[row,:].A[0])
    approx_factor2 = np.asarray(factor2[col,:].A[0])
    approx_entry = np.dot(approx_factor1,approx_factor2)
    #loss on this training example
    tr_loss = loss(actual_entry,approx_entry,loss_type) 

    #update estimates
    fac1_update = gradient(actual_entry, approx_factor2, approx_factor1, loss_type)
    fac1_update += regularization_param * approx_factor1
    factor1[row,:] = approx_factor1 - learning_rate * fac1_update

    fac2_update = gradient(actual_entry, approx_factor1, approx_factor2, loss_type)
    fac2_update += regularization_param * approx_factor2
    factor2[col,:] = approx_factor2 - learning_rate * fac2_update
    
    #NOTE diff took a while
    error = diff(matrix,factor1,factor2)
    if num_iters % 50 == 0:
      errors.append(error)
    num_iters += 1

  print("Found estimate with %f error in %d iterations" % \
    (diff(matrix,factor1,factor2), num_iters))
  print "Errors: ", 
  print errors
  return factor1, factor2

#Compute tolerance criterion
#See if product of factors is close enough (by Frobenius norm)
#to original matrix
#Input: matrix, factors, tolerance
#Output: Frobenius norm of difference between matrix and product 
def diff(matrix,factor1,factor2):
  factor_product_sign = (factor1*factor2.transpose()).sign()
  rows, cols = matrix.nonzero()
  matrix_data = np.asarray(matrix[rows,cols])[0]
  factor_data = np.asarray(factor_product_sign[rows,cols])[0]
  difference = np.sum(matrix_data != factor_data)
  return difference

#Compute gradient of loss function at our point
#with respect to one of the variables
#Input: actual value (label)
#       Vector from component 1
#       Vector from component 2 (take derivative wrt this)
#       loss function type
def gradient(label, comp1, comp2_wrt, loss_type):
  if loss_type == "squarehinge":
    signs = label*np.dot(comp1, comp2_wrt)
    if signs >= 1:
      return np.zeros(comp1.size)
    else:
      grad = 2*(1 - signs)*(-label)*comp1
      return grad
  elif loss_type == "sigmoid":
    sig_loss = sigmoid_loss(label, np.dot(comp1, comp2_wrt))
    grad = sig_loss * (1 - sig_loss) * comp1
    return grad
  else:
    raise ValueError("unrecognized loss function ", loss_type)

#Allow user to choose loss function
#Input: actual and predicted points to compute loss of
#       Loss function type
#Output value: loss from desired loss function
def loss(actual,pred,loss_type):
  if loss_type is "squarehinge":
    return square_hinge_loss(actual,pred)
  if loss_type is "sigmoid":
    return sigmoid_loss(actual,pred)
  else:
    return actual - pred #like in https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf ?

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
  #NOTE: scipy sparse linalg lstsq converts matrices to numpy arrays anyway
  #So we just initialize factors as numpy arrays to save the trouble
  factor1 = np.random.random((matrix.shape[0], dim))
  factor2 = np.random.random((dim, matrix.shape[0]))
  for iteration in range(num_iters):
    #solve 2 least squares problems
    #one fixing the second factor and solving for the first
    #then fix first factor and solve for the second
    factor1 = lstsq(factor2.transpose(), matrix.A)[0]
    factor2 = lstsq(factor1.transpose(), matrix.A)[0]
  return sp.csr_matrix(factor1), sp.csr_matrix(factor2)