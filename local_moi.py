#Contains methods for sign prediction in signed networks
#Local methods: method of influence (balancing cycles)
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
import cPickle


#MOI (measures of imbalance) for link sign prediction
#Predict sign by balancing cycles up to given length
#MOI-\infty is the signed Katz measure
#Input: adjacency matrix
#       dataset name
#       edge to get sign for
#       max cycle order to consider (np.inf for signed Katz)
#       list of discount factors (or single discount factor if inf)
#Output: sign for that edge

#TODO test this code! 
#TODO write LOOCV framework
def predict_sign_MOI(adj_mat, dataset, discount, edge, max_cycle_order):
  if max_cycle_order < 3: #cycle must have length at least 3
    raise ValueError("maximum cycle order must be at least 3")
  if max_cycle_order == np.inf: #compute signed Katz measure
    pass
  else: #compute using formula in Lemma 11 from paper

    #Load in products of adjacency matrix of power up to max cycle order
    #Compute if needed
    products = list()
    current_product = csr_matrix(adj_matrix)
    order = 3
    while order <= max_cycle_order:
      highest_power_product = None
      products_path = "moi_products/product_" + dataset + str(order) + ".npy"
      if os.path.exists(products_path):
        #load this in as highest power product we have so far
        current_product = cPickle.load(products_path).item()
      else:
        current_product = current_product.dot(adj_matrix) #compute next higher power product
        cPickle.dump(current_product, products_path) #...and save it
      products.append(current_product) #add this to our list of products used to compute MOI
      order += 1
    
    #compute imbalance
    imbalance = 0

    #consider up to maximum cycle order
    for cycle_order in range(3,max_cycle_order + 1):
      imbalance += discount_factor[cycle_order] * product[cycle_order][edge]

    #predict and return sign: 1 if imbalance is positive, -1 otherwise
    prediction = 2*(imbalance >= 0) - 1
    return prediction


if __name__ == "__main__":
  data_file_name = "Preprocessed Data/small_network.npy"
  adj_matrix = np.load(data_file_name).item()
  print adj_matrix.nnz
  max_cycle_order = 3
