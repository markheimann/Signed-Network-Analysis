'''Preprocess Epinions and Slashdot data'''
'''Take in text file of data, create and save adjacency matrix'''

import numpy as np, pickle
import scipy.sparse as sp

#Preprocess data
#Optionally run normally or in test mode (when writing tests)
def preprocess(mode = "normal"):
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
    data_matrix = sp.csr_matrix((np.array(labels), (np.array(from_data), 
                  np.array(to_data)) ), shape=(max_id + 1, max_id + 1))

    #correction to make data matrix symmetric
    if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
      data_matrix = (data_matrix + data_matrix.transpose()).sign()

    #test data is a valid symmetric signed matrix
    if mode == "test":
      assert data_matrix.min() == -1
      assert data_matrix.max() == 1
      assert (data_matrix != data_matrix.transpose()).nnz == 0

  #Save data
  np.save("Preprocessed Data/" + dataset + "_csr", data_matrix)

if __name__ == "__main__":
  preprocess()

