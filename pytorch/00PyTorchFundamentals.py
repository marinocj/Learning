import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    #random tensors - important because the way many neural networks is starting by random numbers then
    #adjust them to better represent the data

    #start with random numbers -> look at data -> update numbers -> look at data -> update numbers ....
    
    print( "\n--- Random Tensor ---\n")
    
    #random tensor of size 3,4

    random_tensor = torch.rand(3,4)

    print(random_tensor)
    print(random_tensor.ndim)

    #create a tnesor similar to an image size tensor

    print( "\n--- Random Image Tensor ---\n")

    random_image_size_tensor = torch.rand(size=(3,224,224))
    print(random_image_size_tensor)
    print(random_image_size_tensor.ndim)

    #Create a tensor of all 0s

    print( "\n--- 0 Tensor ---\n")


    zero = torch.zeros(size=(3,4))
    print(zero)
    
    print( "\n--- 0 Tensor ---\n")

    #all ones 

    ones = torch.ones(size=(3,4))
    print(ones)

    print( "\n--- Range of Tensors and Tensors-like ---\n")

    one_to_ten = torch.arange(start=1,end= 11, step=1)
    print(one_to_ten)

    ten_zeros = torch.zeros_like(input=one_to_ten)
    print(ten_zeros)

    ten_ones = torch.ones_like(input=one_to_ten)
    print(ten_ones)
    
    #Tensor Data Types - 3 big errors
    #1 - Tensors not right data type
    #2 - Tensors not right shape
    #3 - Tensors not on the right device


    float_32_tensor = torch.tensor([3.0,6.0,9.0],dtype = None, #datatype (float_32 , float_16)
                                                 device = None, #device (gpu, cpu, cuda)
                                                 requires_grad = False) #whether or not to track gradients with this tneosrs operations
    
    print(float_32_tensor)
    print(float_32_tensor.dtype)

    float_16_tensor = float_32_tensor.type(torch.half) #or torch.float16

    print(float_16_tensor)
    print(float_16_tensor.dtype)

    print()