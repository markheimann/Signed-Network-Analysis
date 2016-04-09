'''Preprocess Epinions and Slashdot data'''
'''Take in text file of data, create and save adjacency matrix'''

import numpy as np, pickle
from scipy.sparse import csr_matrix

FILE_PATH = "Raw Data/soc-sign-epinions.txt"
#FILE_PATH = "Raw Data/soc-sign-Slashdot090221.txt"

#Dataset name (for filename of matrix)
#Split off part right before file extension
dataset = FILE_PATH.split(".txt")[0].split("-")[-1]

with open(FILE_PATH, "rb") as data_file:
  data_lines = data_file.readlines()

  #Save components of data in three lists kept in synchrony
  from_data = list()
  to_data = list()
  labels = list()

  #Data format: each line FROM_ID TO_ID LABEL
  for line_index in range(4, len(data_lines)): #skip first 4 boilerplate lines
    data = data_lines[line_index].split()
    from_data.append(int(data[0]))
    to_data.append(int(data[1]))
    labels.append(int(data[2]))

  #Make a (square) adjacency matrix the size of the number of people 
  #(as given by ID. note: ID starts at 0)

  max_id = max(max(from_data), max(to_data)) #aka number of people

  #Create in sparse row-major format
  #Note: ID starts at 0 so number of people is 1 more than max ID
  data_matrix = csr_matrix((np.array(labels), (np.array(from_data), 
                np.array(to_data)) ), shape=(max_id + 1, max_id + 1))

  #Sanity checks
  print "Number of unique users: ", max_id
  print "Number of edges: ", data_matrix.getnnz()
  print "Matrix shape: ", data_matrix.get_shape()

  #print "max number of times a relationship was voted on: ", max(count_pair_occurrences.values())
  print "min and max values of data matrix: ", data_matrix.min(), data_matrix.max()
  #print "taking signs (equivalently, the mode sign when same relationship was signed multiple times)"
  #data_matrix = data_matrix.sign()
  #print "min and max values of data matrix: ", data_matrix.min(), data_matrix.max()
  print "data matrix is symmetric?", (data_matrix != data_matrix.transpose()).nnz == 0
  if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
    data_matrix = (data_matrix + data_matrix.transpose()).sign()
    print "fix"
    print "data matrix is symmetric?", (data_matrix != data_matrix.transpose()).nnz == 0

  #Save data
  np.save("Preprocessed Data/" + dataset + "_csr", data_matrix)

