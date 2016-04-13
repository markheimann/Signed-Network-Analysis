#Contains methods for sign prediction in signed networks
#Local methods: method of influence (balancing cycles)
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import svds, inv
from scipy.linalg import norm
import cPickle, os


#MOI (measures of imbalance) for link sign prediction
#Predict sign by balancing cycles up to given length
#MOI-\infty is the signed Katz measure
#Input: adjacency matrix
#       dataset name 
#       edge to get sign for
#       max cycle order to consider (np.inf for signed Katz)
#       discount factor (or single discount factor if inf) //list if finite max cycle order, otherwise single value
#Output: sign for that edge
def predict_sign_MOI(adj_mat, dataset, discount_factor, edge, max_cycle_order):
  prediction = None
  if max_cycle_order < 3: #cycle must have length at least 3
    raise ValueError("maximum cycle order must be at least 3")
  if max_cycle_order == np.inf: #compute signed Katz measure
    if type(discount_factor) is not float:
      raise ValueError("discount factor must be float") #TODO must be sufficiently small (< ||A||_2) too? 
    prediction_matrix = identity(adj_matrix.shape[0])
    prediction_matrix = prediction_matrix - discount_factor * adj_matrix
    prediction_matrix = inv(prediction_matrix)
    prediction_matrix = prediction_matrix - identity(adj_matrix.shape[0])
    prediction_matrix = prediction_matrix - discount_factor * adj_matrix

    prediction = 2*(prediction_matrix[edge] >= 0) - 1
  else: #compute using formula in Lemma 11 from paper
    if type(discount_factor) is not list:
      raise ValueError("for finite max cycle order, must provide list of discount factors for each cycle order between 3 and max")


    #Load in products of adjacency matrix of power up to max cycle order
    #Compute if needed
    products = list()
    current_product = csr_matrix(adj_matrix)
    order = 3
    while order <= max_cycle_order:
      highest_power_product = None
      products_path = "moi_products/product_" + dataset + str(order) + ".npy"
      if False: #os.path.exists(products_path): #NOTE can't redo this since adjacency matrix is slightly different for each CV round
        #load this in as highest power product we have so far
        current_product = cPickle.load(open(products_path, "r"))#.item()
      else:
        current_product = current_product.dot(adj_matrix) #compute next higher power product
        #cPickle.dump(current_product, open(products_path, "w")) #...and save it
      products.append(current_product) #add this to our list of products used to compute MOI
      order += 1
    
    #compute imbalance
    imbalance = 0

    #consider up to maximum cycle order
    for cycle_order in range(3,max_cycle_order + 1):
      #subtract 3 for indexing since starting from 3
      imbalance += discount_factor[cycle_order - 3] * products[cycle_order - 3][edge]

    #predict and return sign: 1 if imbalance is positive, -1 otherwise
    prediction = 2*(imbalance >= 0) - 1
  return prediction

#Evaluate MOI with leave-one-out cross-validation
#(Train on all edges except one, test on remaining edge. Rotate through all edges doing this)
#Input: adjacency matrix
#       dataset name (e.g. "small")
#       list of discount factors for each cycle
#       maximum cycle order
#Output: cross-validation accuracy, false positive rate
def loocv_moi(adj_matrix, dataset, discount, max_cycle_order):
  num_data = adj_matrix.nnz
  rows, cols = adj_matrix.nonzero() #each have length num_data

  #get nonzero entries and convert to vector
  data = np.squeeze(np.asarray(adj_matrix[adj_matrix.nonzero()]))

  num_preds = 0 #total predictions
  num_correct = 0 #correct predictions
  num_fp = 0 #false positives
  num_tn = 0 #true negatives
  for datum_index in range(num_data):
    #form edge and get its label
    edge = (rows[datum_index], cols[datum_index])
    edge_label = adj_matrix[edge]

    #create version of adjacency matrix with this edge set to 0
    loo_rows = np.delete(rows, datum_index)
    loo_cols = np.delete(cols, datum_index)
    loo_data = np.delete(data, datum_index)
    loo_adj_matrix = csr_matrix((loo_data, (loo_rows, loo_cols)), shape=adj_matrix.shape)
    predicted_sign = predict_sign_MOI(loo_adj_matrix, dataset, discount, edge, max_cycle_order)
    num_preds += 1 #made a prediction

    if predicted_sign == edge_label: #correct prediction
      num_correct += 1
      if predicted_sign == -1: #correct negative prediction (true negative)
        num_tn += 1
    elif predicted_sign == 1: #false positive
      num_fp += 1

  #compute LOOCV accuracy and false positive rate
  accuracy = float(num_correct)/num_preds
  false_positive_rate = float(num_fp)/(num_fp + num_tn)
  return accuracy, false_positive_rate


if __name__ == "__main__":
  data_file_name = "Preprocessed Data/small_network.npy"
  dataset = "small"
  try:
    adj_matrix = np.load(data_file_name).item()
  except Exception as e:
    raise ValueError("could not load adj matrix from file: ", e)
  #max_cycle_order = 3
  #discount = [0.5**i for i in range(3, max_cycle_order + 1)]
  max_cycle_order = np.inf
  discount = 0.0001
  acc, fpr = loocv_moi(adj_matrix, dataset, discount, max_cycle_order)
  print "Accuracy: ", acc
  print "False positive rate: ", fpr
