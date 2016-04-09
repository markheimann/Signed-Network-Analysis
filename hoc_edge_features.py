#Contains methods for sign prediction in signed networks
#Based on Chiang et. al, 2014

import numpy as np
import cPickle
from scipy.sparse import csr_matrix, find
from scipy.sparse.linalg import svds
from scipy.linalg import norm
import os

#TODO test with small matrices

#Input: adjacency matrix
# Maximum order of cycles to consider
# Name of network being applied to
#Output: List of features for each edge
def extract_edge_features(adj_matrix, max_cycle_order, network_name):
  FEATURES_FNAME = "features_" + network_name + str(max_cycle_order) + ".npy"

  #if we've already computed the features
  if os.path.exists(FEATURES_FNAME): 
  #if False:
    #just load them back in
    print "Loading in precomputed features and labels..."
    feature_dict, labels_dict = cPickle.load(open(FEATURES_FNAME, "r"))

  else: #compute features for the first time and save results
    FEATURE_PRODUCTS_FNAME = "feature_matrices_" + network_name + str(max_cycle_order) + ".npy"
    #already computed matrix products from which features will be extracted
    if os.path.exists(FEATURE_PRODUCTS_FNAME): 
    #if False:
      print "Loading in precomputed feature products..."
      feature_products = cPickle.load(open(FEATURE_PRODUCTS_FNAME,"r"))

    else: #compute feature products for the first time and save results
      print "Calculating feature matrices..."
      print "adjacency matrix is symmetric? ", (adj_matrix != adj_matrix.transpose()).nnz == 0
      print adj_matrix.shape
      #print adj_matrix.todense()[:10,:10]
      #adj_pos_component = csr_matrix(np.zeros(adj_matrix.shape))
      #adj_neg_component = csr_matrix(np.zeros(adj_matrix.shape))
      #adj_pos_component = csr_matrix(adj_matrix, copy=True) #copy by value
      #adj_neg_component = csr_matrix(adj_matrix, copy=True) #copy by value

      #Form positive and negative components of adjacency matrix
      positive_indices = find(adj_matrix == 1)
      negative_indices = find(adj_matrix == -1)
      ones = np.ones(positive_indices[0].size)
      negative_ones = -1 * np.ones(negative_indices[0].size)

      adj_pos_component = csr_matrix((ones, (positive_indices[0], positive_indices[1])), shape = adj_matrix.shape)
      adj_neg_component = csr_matrix((negative_ones, (negative_indices[0], negative_indices[1])), shape = adj_matrix.shape)

    #######NOTE: They say A = A^+ + A^-: would that give negative counts for + * -?
      print "Positive and negative components add to form adjacency matrix? ",
      print (adj_matrix != (adj_pos_component + adj_neg_component)).nnz == 0

      #print adj_pos_component.todense()[:10,:10]

      #For easy indexing of which component we want to work with
      components = (adj_pos_component, adj_neg_component)#,
                    #adj_pos_component.transpose(), adj_neg_component.transpose())
      print "adj matrix pos component is symmetric?", (adj_pos_component != adj_pos_component.transpose()).nnz == 0
      print "adj matrix neg component is symmetric?", (adj_neg_component != adj_neg_component.transpose()).nnz == 0
      #NOTE: we are considering only undirected graphs so transpose is the same

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
  if max_length >= max_cycle_order: #have achieved all instructions of desired length
    return products

  #recursively grow set of instructions
  #print "products: ", products[max_length]
  new_products = list()
  for product in products[max_length]:
    #print "product: ", product
    for new_product_term in components:
      new_product = csr_matrix(product) #shallow copy of product
      new_product = new_product.dot(new_product_term) #multiply by another component
      new_products.append(new_product)
  products[max_length + 1] = new_products #next order higher matrix features
  return compute_feature_products(components, max_cycle_order, products)

#Extract features from feature matrices for a given edge
#Input: feature matrices
def extract_features_for_edge(feature_matrices, from_vertex, to_vertex):
  features = list()
  for key in feature_matrices.keys():
    for product in feature_matrices[key]:
      features.append(product[from_vertex,to_vertex])
  return features


if __name__ == "__main__":
  #TODO write formal test but I think this is right
  matrix1 = csr_matrix(np.asarray([[2,3],[4,5]]))
  matrix2 = csr_matrix(np.asarray([[1,6],[3,2]]))
  matrix3 = csr_matrix(np.asarray([[4,4],[5,2]]))
  matrix4 = csr_matrix(np.asarray([[3,2],[8,-1]]))
  components = [matrix1, matrix2, matrix3, matrix4]
  products = compute_feature_products(components,3)
  for length in products.keys():
    print np.array_equal(products[length][0].todense(), (matrix1 ** length).todense())
