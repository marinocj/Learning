import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #Manipulating tensors
    #Operation include addition, 
    # - subtraction, 
    # - multiplication, 
    # - division, 
    # - matrix multiplication
    print("\n---Addition---\n")
    tensor = torch.tensor([1,2,3])
    print(tensor)
    tensor = tensor + 10
    print(tensor)

    print("\n---Multiplication---\n")
    
    tensor = torch.tensor([1,2,3])
    print(tensor)
    tensor = tensor * 10
    print(tensor)

    print("\n---Subtraction---\n")
    
    tensor = torch.tensor([10,20,30])
    print(tensor)
    tensor = tensor - 1
    print(tensor)

    print("\n---Division---\n")
    
    tensor = torch.tensor([10,20,30])
    print(tensor)
    tensor = tensor / 10
    print(tensor)

    #Matrix multiplication - probably the most common 

    print("\n---Matrix Multiplication---\n")

    tensor = torch.tensor([1,2,3])
    print(tensor)
    print(tensor*tensor)

    tensor = torch.tensor([1,2,3])
    print(tensor)
    print(torch.matmul(tensor,tensor))

    #matrix multiplication by hand
    # (1*1) + (2*2) + (3*3) 
      
    #rules for larger matricies
    print()