#Contains methods for sign prediction in signed networks
#Local methods: method of influence (balancing cycles)
#Based on Chiang et. al, 2014

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import norm


#Compute
#Input: adjacency matrix, two vertices defining edge
# Maximum order of cycles to consider
#Output: Sign prediction of edge
def compute_cycle_balances(from_vertex, to_vertex, adj_matrix, max_cycle_order):
  cycles = get_all_cycles_including_edge(from_vertex, to_vertex, adj_matrix, max_cycle_order)
  #balanced cycle: even number of negative edges (product of signs, not counting 0s for unknown, is 1)
  #Algorithm: BFS (look for all paths from "from_vertex" to "to_vertex" of given length up to max_cycle_order)
  #TODO: how to tell apart unknown edges from nonexistent edges?  

#Get all cycles up to maximum length including an edge
#Input: adjacency matrix, two vertices defining edge
# Maximum order of cycles to consider
def get_all_cycles_including_edge(from_vertex, to_vertex, adj_matrix, max_cycle_order):
  pass
  #Algorithm: find all paths from "to_vertex" to "from_vertex" using BFS (cycles from "from_vertex")
  #find all paths from "from_vertex" to "to_vertex" using BFS (cycles from "to_vertex")

if __name__ == "__main__":
  pass
