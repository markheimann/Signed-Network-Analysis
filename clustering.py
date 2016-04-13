#Clustering algorithms for signed networks
#Based on Chiang et. al, 2014

from scipy.sparse import csr_matrix, diags
from scipy.linalg import eig
import numpy as np

import matrix_factorization as mf
from sklearn.cluster import KMeans
import simulate_networks as sim

#Perform matrix clustering
#Try to find clusterings such that within-cluster edges are 1 and between-cluster edges are -1
#Input: adjacency matrix
#       number of clusters
#       method to use ("signed laplacian" from previous work, or "matrix completion" from this paper)
#       completion algorithm (if doing matrix completion)
#       params for completion algorithm (if doing matrix completion)
#Output: vector of cluster indicators for each row/column (person) in adjacency matrix
def cluster_signed_network(adj_matrix, num_clusters, method = "signed laplacian", completion_alg=None, params=None):
  cluster_matrix = None #matrix whose eigenvectors to perform clustering on top of
  method = method.lower()

  if method == "signed laplacian":
    sl_diag_data = list()
    for row_index in range(adj_matrix.shape[0]):
      row = adj_matrix.getrow(row_index).A[0]
      #sum absolute values of all non-diagonal row elements
      sl_diag_data.append(sum([abs(row_entry) for row_entry in row if row_entry != row_index]))
    sl_diag_matrix = diags([sl_diag_data],[0],format="csr") #make diagonal matrix with this data
    cluster_matrix = sl_diag_matrix - adj_matrix #signed Laplacian
  elif method == "matrix completion":
    completion_alg == completion_alg.lower()

    #Complete the matrix with desired algorithm
    if completion_alg == "svp":
      try:
        rank, tol, max_iter, step_size = params
        cluster_matrix = svp.sign_prediction_SVP(adj_matrix, rank, tol, max_iter, step_size)
      except:
        raise ValueError("check input for SVP")
    elif completion_alg == "sgd":
      try:
        learning_rate, loss_type, tol, max_iters, regularization_param, dim = params
        factor1, factor2 = mf.matrix_factor_SGD(adj_matrix, learning_rate, loss_type, tol, max_iters, regularization_param, dim)
        cluster_matrix_matrix = csr_matrix.sign(csr_matrix(factor1*factor2.transpose()))
      except:
        raise ValueError("check input for SGD")
    elif completion_alg == "als":
      try: 
        dim, num_iters = params
        factor1, factor2 = mf.matrix_factor_ALS(adj_matrix, dim, num_iters)
        cluster_matrix = csr_matrix.sign(csr_matrix(factor1.transpose()*factor2))
      except:
        raise ValueError("check input for ALS")
    else:
      raise ValueError("unrecognized matrix completion algorithm: ", completion_alg)
  else:
    raise ValueError("unrecognized clustering method: ", method)

  #Get top k (number of clusters) eigenvectors
  #CONFIRM right eigenvectors, correct?
  eigenvals, r_eigenvecs = eig(cluster_matrix.A)
  top_eigenvecs = r_eigenvecs[:,0:num_clusters]
  assert top_eigenvecs.shape == (adj_matrix.shape[0], num_clusters)

  #Perform k-means clustering on top k eigenvectors
  clusters = KMeans(n_clusters = num_clusters)
  cluster_indicators = clusters.fit_predict(top_eigenvecs)
  assert cluster_indicators.size == adj_matrix.shape[0] #predicted one cluster for each person
  return cluster_indicators

#Clustering pipeline
#Simulate data, store ground truth clusters
#Perform clustering algorithm, get predicted clusters
#Compute clustering accuracy
#Input: tuple of network parameters (see simulate_networks.py for details)
#       tuple of clustering parameters (see cluster_signed_network() for details)
#Output: predictions of cluster label (0-num_clusters - 1) for each label
def clustering_pipeline(network_params, completion_params):
  #Create network
  num_clusters, cluster_size, sparsity, noise_prob = network_params
  
  network = sim.sample_network(num_clusters, cluster_size, sparsity, noise_prob)
  
  #Assign ground truth labels (the first "cluster_size" are in cluster 0, next are in cluster 1, etc.)
  cluster_labels = list()
  for cluster_num in range(num_clusters):
    cluster_labels += [cluster_num] * cluster_size
  cluster_labels = np.asarray(cluster_labels)

  #perform clustering
  method, completion_alg, completion_params = clustering_params  
  cluster_predictions = cluster_signed_network(network, num_clusters, method, completion_alg, completion_params)

  print "Accuracy: ", np.mean(cluster_predictions == cluster_labels)

if __name__ == "__main__":
  #Parameters for simulating network
  num_clusters = 5
  cluster_size = 10
  sparsity = 0.25
  noise_prob = 0.25
  network_params = (num_clusters, cluster_size, sparsity, noise_prob)

  #Parameters for performing clustering
  '''
  method = "signed laplacian"
  completion_alg = None
  completion_params = None
  '''

  method = "matrix completion"
  completion_alg = "als"
  dim = num_clusters*cluster_size/4
  num_iters = 5
  completion_params = (dim, num_iters)
  clustering_params = (method, completion_alg, completion_params)
  clustering_pipeline(network_params, clustering_params)




  