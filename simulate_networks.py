#Simulate signed networks
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm
from scipy.stats import powerlaw

#Input: number and size of clusters
#   sparsity level (percentage of edges to sample)
#   noise level (probability sample edge sign is flipped)
#   sampling process (uniform or power law)
#Output: noisy sampled network
#TODO: write official tests
#TODO: power law sampling
def sample_network(num_clusters,
                   cluster_size, 
                   sparsity_level, 
                   noise_prob, 
                   sampling_process = "uniform"):

  #will fill in with correct entries
  #sampled_network = np.zeros(full_network.shape).flatten() #sampling from 1D array is easier
  network_dimension = num_clusters * cluster_size
  network_size = network_dimension**2
  num_samples = sparsity_level*network_size #number of samples to take from network

  #Get probability with which to sample each entry
  probs = None #uniform (default choice)
  #implement power law sampling at some point? (Chung-Lu-Vu random graph model?)
  if sampling_process == "power_law":
    raise ValueError("power law sampling not implemented")

  #Sample indices
  samples = np.random.choice(np.arange(network_size), num_samples, replace=False, p = probs)
  
  #Determine which indices to sample noisily (with wrong sign)
  
  edges = [get_edge_from_index(sample, network_dimension) for sample in samples]
  rows, cols = zip(*edges)

  #Compute true data
  data = np.asarray([get_complete_balanced_network_edge_sign(cluster_size, edge) for edge in edges])
  
  #Make data noisy: 1s are true, -1s are noise
  noise = 2 * (np.random.random(num_samples) > noise_prob) - 1 #1 if entry should be sampled correctly, -1 otherwise
  data = data * noise

  sampled_network = csr_matrix((data, (rows, cols)), shape = (network_dimension, network_dimension))
  return sampled_network

#Map a number between 1 and n
#to a 2-d coordinate with entries between 1 and sqrt(n)
#That way we can sample 2D edges from a 1D array
#Basically, we are implicitly un-flattening an array

#Input: index to convert to 2D
#       matrix dimension size (num rows/cols in square matrix)
#Output: 2D coordinate to which that index maps
def get_edge_from_index(index, matrix_dim_size):
  row_index = index / matrix_dim_size
  col_index = index % matrix_dim_size
  return (row_index, col_index)

#Implicitly create a complete k-weakly balanced network
#   within cluster edges positive, between cluster edges negative
#Avoid explicitly creating the complete matrix in case it's very big
#In paper set k = 5 with 100 nodes per cluster
#   how many nodes to each cluster
#   edge to get sign of
#Output: sign of desired edge
def get_complete_balanced_network_edge_sign(cluster_size, edge):
  #note: balanced network is basically blocks of 1s on the diagonal, -1s everywhere else
  row_index, col_index = edge
  #edges in same cluster if they are the same multiple of cluster_size
  #(not counting remainder, which only specifies their location in the block)
  return 2*(col_index/cluster_size == row_index/cluster_size) - 1 


if __name__ == "__main__":
  #TODO write test
  cluster_size = 2
  num_clusters = 5
  network_size = num_clusters*cluster_size
  sparsity = 0.9
  noise_prob = 0.9
  sim_full_network = np.asarray([get_complete_balanced_network_edge_sign(cluster_size,get_edge_from_index(edge,network_size)) for edge in np.arange(network_size*network_size)]).reshape(network_size,network_size)
  sim_partial_network = sample_network(num_clusters,cluster_size,sparsity,noise_prob)
  print "Full network: "
  print sim_full_network
  print "Partial network: "
  print sim_partial_network.todense()
  print sim_full_network == sim_partial_network
  print sim_full_network == -1 * sim_partial_network

