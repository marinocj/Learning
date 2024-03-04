import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)

if __name__ == "__main__":
    #intro to tensors

    #scalar tensor
    #Ptorch tensors are created using torch.tensor()

    print( "\n--- Scalar ---\n")
    scalar = torch.tensor(7)
    print(scalar)
    print(scalar.ndimension())

    # vector 

    print( "\n--- Scalar ---\n")
    vector = torch.tensor([7, 7])
    print(vector)
    print(vector.ndimension())
    print(vector.size())

    #matrix

    print( "\n--- Matrix ---\n")
    MATRIX = torch.tensor([ [7, 8] , [9, 10] ])
    print(MATRIX)
    print(MATRIX.ndim)
    print(MATRIX[0])
    print(MATRIX[1])
    print(MATRIX.shape)

    #TENSOR

    print( "\n--- Tensor ---\n")
    TENSOR = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
    print(TENSOR.shape)
    print(TENSOR[0])


    print("\n")





    



