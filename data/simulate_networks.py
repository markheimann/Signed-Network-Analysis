#Simulate signed networks
#Based on Chiang et. al, 2014
import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
from scipy.stats import powerlaw

#Input: list of sizes of each cluster [list]
#   sparsity level (percentage of edges to sample) [float 0-1]
#   noise level (probability sample edge sign is flipped) [float 0-1]
#   whether or not to make the matrix symmetric [boolean]
#   sampling process [str "uniform" or "power_law"]
#Output: noisy sampled network [sparse csr matrix]
#(optional) TODO: power law sampling
def sample_network(cluster_sizes, 
                   sparsity_level, 
                   noise_prob, 
                   symmetric = True,
                   sampling_process = "uniform"):

  #will fill in with correct entries
  network_dimension = sum(cluster_sizes)

  #network_dimension = num_clusters * cluster_size
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
  data = np.asarray([get_complete_balanced_network_edge_sign(cluster_sizes, edge) for edge in edges])
  
  #Make data noisy: 1s are true, -1s are noise
  #1 if entry should be sampled correctly, -1 otherwise
  noise = 2 * (np.random.random(num_samples) > noise_prob) - 1
  data = data * noise

  sampled_network = sp.csr_matrix((data, (rows, cols)), shape = (network_dimension, network_dimension))
  if symmetric:
    sampled_network = (sampled_network + sampled_network.transpose()).sign() #make symmetric
  return sampled_network

#Map a number between 1 and n^2
#to a 2-d coordinate with entries between 1 and n
#That way we can sample 2D edges from a 1D array
#Basically, we are implicitly un-flattening an array

#Input: index to convert to 2D [int]
#       matrix dimension size (num rows/cols in square matrix) [int]
#Output: 2D coordinate to which that index maps [tuple of 2 ints]
def get_edge_from_index(index, matrix_dim_size):
  if index < 0 or index >= matrix_dim_size**2:
    raise ValueError("index out of matrix range")
  row_index = index / matrix_dim_size
  col_index = index % matrix_dim_size
  return (row_index, col_index)

#Implicitly create a complete k-weakly balanced network
#   within cluster edges positive, between cluster edges negative
#Avoid explicitly creating the complete matrix in case it's very big
#In paper set k = 5 with 100 nodes per cluster
#   how many nodes to each cluster [list of ints]
#   edge to get sign of [2-tuple of ints]
#Output: sign of desired edge
def get_complete_balanced_network_edge_sign(cluster_sizes, edge):
  #note: balanced network is basically blocks of 1s on the diagonal, -1s everywhere else
  row_index, col_index = edge

  #find which cluster edge would be in
  #(i.e. for edge to be in a cluster, row and col are in same cluster range)
  sign = -1 #assume edge isn't in a cluster by default
  #go through each clusters and see if they're in the same cluster
  num_considered = 0
  for cluster_size in cluster_sizes:
    row_in_cluster = False
    col_in_cluster = False
    if (row_index >= num_considered and row_index < num_considered + cluster_size):
      row_in_cluster = True
    if (col_index >= num_considered and col_index < num_considered + cluster_size):
      col_in_cluster = True
    if row_in_cluster and col_in_cluster: #we've found edge and in cluster
      sign = 1
      break
    #exclusive or: row is in this cluster range but column isn't (so edge not in a cluster)
    elif row_in_cluster ^ col_in_cluster: #we've found edge and not in cluster
      break #leave sign -1
    else:
      num_considered += cluster_size #keep looking for edge
  return sign

#Construct a complete network explicitly
#Mainly used for small examples for testing
#Input: cluster sizes [list of ints]
#Output: adjacency matrix of complete network [sparse csr matrix]
def construct_full_network(cluster_sizes):
  network_size = sum(cluster_sizes)

  #list of signs of edges
  sim_full_network_edges_list = list()

  #compute appropriate sign of each edge and save it in list
  for edge_number in range(network_size**2):
    edge = get_edge_from_index(edge_number,network_size)
    edge_sign = get_complete_balanced_network_edge_sign(cluster_sizes,edge)
    sim_full_network_edges_list.append(edge_sign)

  #convert to numpy array
  np_full_network_edges = np.asarray(sim_full_network_edges_list)
  #give appropriate shape
  np_full_network = np_full_network_edges.reshape(network_size,network_size)
  #convert to CSR matrix
  sim_full_network = sp.csr_matrix(np_full_network)
  return sim_full_network
