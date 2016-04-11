#Local method: use machine learning classifier to predict edge sign
#using features of that edge extracted from "higher order cycles" (HOC)
#This file implements methods to extract those features from the graph
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.sparse import csr_matrix, find
from scipy.sparse.linalg import svds
from scipy.linalg import norm
import os


#Input: adjacency matrix
# Maximum order of cycles to consider
# Name of network being applied to
# Mode: can run normally or in test mode (used for unit tests)
#Output: List of features for each edge
def extract_edge_features(adj_matrix, network_name, max_cycle_order, mode="normal"):
  if max_cycle_order < 3: #cycles must be at least length 3
    raise ValueError("Cycles must have length at least 3")
  FEATURES_FNAME = "hoc_features/features/features_" + network_name + str(max_cycle_order) + ".npy"

  #if we've already computed the features
  if os.path.exists(FEATURES_FNAME): 
  #if False: #left in here in case we want to compute everything from scratch (e.g. to measure run time)
    #just load them back in
    print "Loading in precomputed features and labels..."
    feature_dict, labels_dict = cPickle.load(open(FEATURES_FNAME, "r"))

  else: #compute features for the first time and save results
    FEATURE_PRODUCTS_FNAME = "hoc_features/feature_matrices/feature_matrices_" + network_name + str(max_cycle_order) + ".npy"
    #already computed matrix products from which features will be extracted
    if os.path.exists(FEATURE_PRODUCTS_FNAME): 
    #if False: #left in here in case we want to compute everything from scratch (e.g. to measure run time)
      print "Loading in precomputed feature products..."
      feature_products = cPickle.load(open(FEATURE_PRODUCTS_FNAME,"r"))

    else: #compute feature products for the first time and save results
      print "Calculating feature matrices..."
      #Check for symmetry of adjacency matrix
      #NOTE: these techniques could be adapted to handle non-symmetric adjacency matrices
      #but that's outside the scope of this project (maybe do at some later point)
      if (adj_matrix != adj_matrix.transpose()).nnz > 0:
        raise ValueError("adjacency matrix is not symmetric")

      #Form positive and negative components of adjacency matrix
      positive_indices = find(adj_matrix == 1)
      negative_indices = find(adj_matrix == -1)
      ones = np.ones(positive_indices[0].size)
      negative_ones = -1 * np.ones(negative_indices[0].size)
      adj_pos_component = csr_matrix((ones, (positive_indices[0], positive_indices[1])), shape = adj_matrix.shape)
      adj_neg_component = csr_matrix((negative_ones, (negative_indices[0], negative_indices[1])), shape = adj_matrix.shape)

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
      feature_products = compute_feature_products(components, max_cycle_order)

      #save computation results so don't have to do them again
      cPickle.dump(feature_products, open(FEATURE_PRODUCTS_FNAME, "w"))

    print "Extracting features from feature matrices..."
    feature_dict = dict()
    labels_dict = dict()

    #indices of rows and columns to get features for
    edge_row_indices, edge_col_indices = adj_matrix.nonzero()

    #(i,j) and (j,i) have same features--model only considers symmetric relationships
    #only get features for (i,j)
    unique_edges = set()
    for data_index in range(len(edge_row_indices)): #go through all data points
      edge = (edge_row_indices[data_index], edge_col_indices[data_index])
      reverse_edge = edge[::-1]
      #if we haven't already seen edge (i,j) or (j,i)
      if edge not in unique_edges and reverse_edge not in unique_edges:
        unique_edges.add(edge) #now we've seen it
        edge_features = extract_features_for_edge(feature_products, edge_row_indices[data_index], edge_col_indices[data_index])
        feature_dict[edge] = edge_features
        label = adj_matrix[edge]
        labels_dict[edge] = label

    #TEST: dictionary formed properly (no duplicate edges)
    if mode == "test":
      #print feature_dict.keys()
      for edge in feature_dict.keys(): #exactly same edges added to label dict so only test one
        i,j = edge
        assert (j,i) not in feature_dict or j == i #OK if edge is self loop (we know person's self opinion? :P)

        #make sure we have correct number of features
        assert len(feature_dict[edge]) == sum([2**x for x in range(2,max_cycle_order)])

    cPickle.dump((feature_dict, labels_dict), open(FEATURES_FNAME, "w"))

  return feature_dict, labels_dict

#Multiply the matrices for which the i,j-th entry will be a feature for the edge (i,j)
#Input: positive and negative components of adjacency matrix + their transposes
#   maximum cycle order
#Output: matrices whose i,j-th entry is a feature for the edge i,j
def compute_feature_products(components, max_cycle_order, products = None):
  if products is None: #first call (not recursive)
    products = {1: [component for component in components]}
  max_length = max(products.keys()) #maximum length instructions generated so far
  #note: max product length is 2 for max cycle order of 3 (3-cycle info given by matrix (\pm)A^(T)*(\pm)A) )
  if max_length >= max_cycle_order - 1: #have achieved all instructions of desired length
    return products

  #recursively grow set of instructions
  new_products = list()
  for product in products[max_length]:
    for new_product_term in components:
      new_product = csr_matrix(product) #shallow copy of product
      new_product = new_product.dot(new_product_term) #multiply by another component
      new_products.append(new_product)
  products[max_length + 1] = new_products #next order higher matrix features
  return compute_feature_products(components, max_cycle_order, products)

#Extract features from feature matrices for a given edge
#Input: feature matrices
#Output: features for that edge
def extract_features_for_edge(feature_matrices, from_vertex, to_vertex):
  features = list()
  for key in feature_matrices.keys():
    #don't count products of length 1, which includes just the original matrix itself
    #then one of the features will be the actual label
    if key == 1:
      continue 

    for product in feature_matrices[key]:
      features.append(product[from_vertex,to_vertex])
  return features
