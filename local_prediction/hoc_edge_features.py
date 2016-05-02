#Local method: use machine learning classifier to predict edge sign
#using features of that edge extracted from "higher order cycles" (HOC)
#This file implements methods to extract those features from the graph
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.linalg import norm
import os

#Input: adjacency matrix
# Maximum order of cycles to consider
# Mode: can run normally or in test mode (used for unit tests)
#Output: List of features for each edge
def extract_edge_features(adj_matrix, max_cycle_order, mode="normal"):
  if max_cycle_order < 3: #cycles must be at least length 3
    raise ValueError("Cycles must have length at least 3")
  print "Calculating feature matrices..."
  #Check for symmetry of adjacency matrix
  #NOTE: these techniques could be adapted to handle non-symmetric adjacency matrices
  #but that's outside the scope of this project (maybe do at some later point)
  if (adj_matrix != adj_matrix.transpose()).nnz > 0:
    raise ValueError("adjacency matrix is not symmetric")

  #Form positive and negative components of adjacency matrix
  positive_indices = sp.find(adj_matrix == 1)
  negative_indices = sp.find(adj_matrix == -1)
  ones = np.ones(positive_indices[0].size)
  negative_ones = -1 * np.ones(negative_indices[0].size)
  adj_pos_component = sp.csr_matrix((ones, (positive_indices[0], positive_indices[1])), shape = adj_matrix.shape)
  adj_neg_component = sp.csr_matrix((negative_ones, (negative_indices[0], negative_indices[1])), shape = adj_matrix.shape)

  #For easy indexing of which component we want to work with
  #NOTE: we are considering only undirected graphs so transpose is the same 
  #(unlike Appendix B of the paper which describes feature computation for directed graphs)
  components = (adj_pos_component, adj_neg_component)

  #TEST: components of adjacency matrix are in fact formed properly
  if mode == "test":
    #make sure positive and negative components are symmetric 
    #(they are, unless programming bug, if adjacency matrix is)
    assert (adj_pos_component != adj_pos_component.transpose()).nnz == 0
    assert (adj_neg_component != adj_neg_component.transpose()).nnz == 0

    #make sure positive and negative components add to form adjacency matrix
    assert (adj_matrix != (adj_pos_component + adj_neg_component)).nnz == 0

  #perform matrix multiplications to compute features
  #features will be the (i,j)-th entry of each product

  #first see if we've calculated any lower order products that
  #if so, use these calculations instead of recomputing on the way
  #to calculating higher-order products
  products = None
  print "computing feature products..."
  feature_products = compute_feature_products(components, max_cycle_order, products)
  return feature_products

#Multiply the matrices for which the i,j-th entry will be a feature for the edge (i,j)
#Input: positive and negative components of adjacency matrix + their transposes
#   maximum cycle order
#Output: matrices whose i,j-th entry is a feature for the edge i,j
def compute_feature_products(components, max_cycle_order, products = None):
  max_components = None
  max_length = 1
  if products is None: #first call (not recursive)
    products = dict() #make dictionary
    max_components = components
  else:
    max_length = max(products.keys()) #max length instructions computed so far
    max_components = products[max_length]
  #note: max product length is 2 for max cycle order of 3 (3-cycle info given by matrix (\pm)A^(T)*(\pm)A) )
  if max_length >= max_cycle_order - 1: #have achieved all instructions of desired length
    return products

  #recursively grow set of instructions
  new_products = list()
  for product in max_components: #products[max_length]:
    for new_product_term in components:
      new_product = sp.csr_matrix(product) #shallow copy of product
      new_product = new_product.dot(new_product_term) #multiply by another component
      new_products.append(new_product)
  products[max_length + 1] = new_products #next order higher matrix features
  return compute_feature_products(components, max_cycle_order, products)

#Extract features from feature matrices for a given edge
#Input: feature matrices
#       edge to get features for
#Output: features for that edge
def extract_features_for_edge(feature_matrices, edge):
  from_vertex, to_vertex = edge
  features = list()
  for key in feature_matrices.keys():
    for product in feature_matrices[key]:
      features.append(product[from_vertex,to_vertex])
  return features

if __name__ == "__main__":
  prods = compute_feature_products([np.random.random((2,2)),np.random.random((2,2))], 5)
  for key in prods.keys():
    print len(prods[key])
