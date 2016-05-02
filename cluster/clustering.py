#Clustering algorithms for signed networks
#Based on Chiang et. al, 2014

import scipy.sparse as sp
from scipy.linalg import eig, cholesky, LinAlgError
import numpy as np
import logging, time

from sklearn.cluster import KMeans

import matrix_completion.svp_sign_prediction as svp
import matrix_completion.matrix_factorization as mf
import data.simulate_networks as sim
import kernel.graph_kernel as kernel
import analytics.stats as stats

from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError

#Perform matrix clustering
#Try to find clusterings such that within-cluster edges are 1
# and between-cluster edges are -1
#Input: adjacency matrix [scipy sparse csr matrix]
#       number of clusters [int]
#       method to use [str: "signed laplacian", "matrix completion"]
#       completion algorithm [str] (if doing matrix completion)
#       params for completion algorithm (if doing matrix completion)
#         [tuple of all parameters for completion algorithm besides adj matrix]
#Output: vector of cluster indicators for each row/col in adjacency matrix [np array]
def cluster_signed_network(adj_matrix, 
                            cluster_sizes, 
                            method, 
                            completion_alg=None, 
                            params=None, mode="normal"):
  cluster_matrix = None #matrix whose eigenvectors to perform clustering on top of
  num_clusters = len(cluster_sizes)
  method = method.lower()

  if method == "signed laplacian":
    cluster_matrix = kernel.signed_laplacian(adj_matrix)
    if mode == "test": #test to make sure signed Laplacian is positive semidefinite
      try:
        chol = cholesky(cluster_matrix.A)
      except LinAlgError:
        raise ValueError("Cholesky failed, so signed Laplacian is not PSD...")
  elif method == "matrix completion":
    completion_alg == completion_alg.lower()

    #Complete the matrix with desired algorithm
    if completion_alg == "svp":
      try:
        rank, tol, max_iter, step_size = params
        cluster_matrix = svp.sign_prediction_SVP(adj_matrix, rank, tol, max_iter, step_size)
      except:
        logging.exception("Exception: ")
        raise ValueError("check input for SVP")
    elif completion_alg == "sgd":
      try:
        learning_rate, loss_type, tol, max_iters, regularization_param, dim = params
        factor1, factor2 = mf.matrix_factor_SGD(adj_matrix, learning_rate, \
                                loss_type, tol, max_iters, regularization_param, dim)
        cluster_matrix = sp.csr_matrix.sign(sp.csr_matrix(factor1*factor2.transpose()))
      except:
        raise ValueError("check input for SGD")
    elif completion_alg == "als":
      try: 
        dim, num_iters = params
        factor1, factor2 = mf.matrix_factor_ALS(adj_matrix, dim, num_iters)
        cluster_matrix = sp.csr_matrix.sign(sp.csr_matrix(factor1.transpose()*factor2))
      except:
        logging.exception()
        raise ValueError("check input for ALS")
    else:
      raise ValueError("unrecognized matrix completion algorithm: ", completion_alg)

    #see how much of underlying complete matrix our completion method recovered
    rows, cols = cluster_matrix.nonzero()
    edges = zip(rows, cols)
    num_correct_edges = 0
    for edge in edges:
      if cluster_matrix[edge] == sim.get_complete_balanced_network_edge_sign(cluster_sizes,edge):
        num_correct_edges += 1
    print "percentage of network recovered: ", float(num_correct_edges) / sum(cluster_sizes)**2
  else:
    raise ValueError("unrecognized clustering method: ", method)

  #Get top k (number of clusters) eigenvectors
  print "getting eigenvalues"
  print cluster_matrix.shape
  cluster_matrix = cluster_matrix.asfptype()
  try:
    eigenvals, top_eigenvecs = sp.linalg.eigs(cluster_matrix, k=num_clusters)
  #happens when adj matrix is complete sparse matrix with 5 clusters of size 5(???)
  except ArpackError:
    eigenvals, eigenvecs = eig(cluster_matrix.A)
    top_eigenvecs = eigenvecs[:,:num_clusters]
  if mode == "test":
    assert top_eigenvecs.shape == (adj_matrix.shape[0], num_clusters)

  #Perform k-means clustering on top k eigenvectors
  clusters = KMeans(n_clusters = num_clusters)
  cluster_indicators = clusters.fit_predict(top_eigenvecs)

  #test: predicted one cluster for each person
  if mode == "test":
    assert cluster_indicators.size == adj_matrix.shape[0]
  return cluster_indicators

#Clustering pipeline
#Simulate data, store ground truth clusters
#Perform clustering algorithm, get predicted clusters
#Compute clustering accuracy
#Input: tuple of network parameters (see simulate_networks.py for details)
#       tuple of clustering parameters (see cluster_signed_network() for details)
#Output: predictions of cluster label (0-num_clusters - 1) for each label
def clustering_pipeline(network_params, clustering_params):
  #Create network
  cluster_sizes, sparsity, noise_prob = network_params
  num_clusters = len(cluster_sizes)
  
  network = sim.sample_network(cluster_sizes, sparsity, noise_prob)
  rows, cols = network.nonzero()
  
  #Assign ground truth labels (the first "cluster_size" are in cluster 0, 
  #next are in cluster 1, etc.)
  cluster_labels = list()
  for cluster_index in range(len(cluster_sizes)):
    cluster_labels += [cluster_index] * cluster_sizes[cluster_index]
  cluster_labels = np.asarray(cluster_labels)

  #perform clustering
  cluster_sizes, method, completion_alg, completion_params, mode = clustering_params  
  cluster_predictions = cluster_signed_network(network, cluster_sizes, method, \
                            completion_alg, completion_params, mode)
  cluster_accuracy = evaluate_cluster_accuracy(cluster_predictions, cluster_labels, \
                            rows, cols)
  return cluster_accuracy

#Evaluate clustering accuracy
#Input: predicted cluster labels [np array]
#       ground truth cluster assignments [np array]
#       rows, cols of edges in the graph [both lists]
#Output: proportion of correctly clustered
def evaluate_cluster_accuracy(cluster_predictions, cluster_labels, rows, cols):
  num_correctly_clustered = 0
  num_data = len(rows)
  for edge_index in range(num_data):
    row = rows[edge_index]
    col = cols[edge_index]

    #edges are in the same cluster (both predicted and actually)
    if cluster_predictions[row] == cluster_predictions[col] and \
          cluster_labels[row] == cluster_labels[col]:
      num_correctly_clustered += 1

    #edges aren't in the same cluster (both predicted and actually)
    elif cluster_predictions[row] != cluster_predictions[col] and \
          cluster_labels[row] != cluster_labels[col]:
      num_correctly_clustered += 1
  return float(num_correctly_clustered)/num_data


if __name__ == "__main__":
  use_signed_laplacian = True
  use_mf = True

  #Parameters for simulating network
  cluster_sizes = [100]*10
  sparsity = 0.05
  noise_prob = 0.01
  network_params = (cluster_sizes, sparsity, noise_prob)
  mode="normal" #"test"
  #Parameters for performing clustering

  if use_signed_laplacian:
  
    sl_method = "signed laplacian"
    sl_completion_alg = None
    sl_completion_params = None

    sl_clustering_params = (cluster_sizes, sl_method, sl_completion_alg, \
                              sl_completion_params, mode)
    before_sl_cluster = time.time()
    sl_cluster_accuracy = clustering_pipeline(network_params, sl_clustering_params)
    after_sl_cluster = time.time()
    print "Clustering results with signed Laplacian: "
    print "Clustering accuracy: ", sl_cluster_accuracy
    print "Clustering standard error: ", 
    print stats.error_width(stats.sample_std(sl_cluster_accuracy), sum(cluster_sizes))
    print "Clustering running time: ", after_sl_cluster - before_sl_cluster
  if use_mf:
    mf_method = "matrix completion"
    mf_completion_alg = "svp"
    rank = 40
    tol = 100
    max_iter = 5
    step_size = 1
    mf_completion_params = (rank, tol, max_iter, step_size)
    
    mf_clustering_params = (cluster_sizes, mf_method, mf_completion_alg, mf_completion_params, mode)
    before_mf_cluster = time.time()
    mf_cluster_accuracy = clustering_pipeline(network_params, mf_clustering_params)
    after_mf_cluster = time.time()
    print "Clustering results with matrix completion: "
    print "Clustering accuracy: ", mf_cluster_accuracy
    print "Clustering standard error: ",
    print stats.error_width(stats.sample_std(mf_cluster_accuracy), sum(cluster_sizes))
    print "Clustering running time: ", after_mf_cluster - before_mf_cluster




  