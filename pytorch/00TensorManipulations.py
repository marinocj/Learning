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
    #some of the most common errors are shape errors
    #the inner dimesions must match
    # (3,2) @ (3,2) - error
    # (2,3) @ (3,2) - works 
    # (3,2) @ (2,3) - works

    #the resulting matix will be the shape of the outer dimensions
    # (2,3) @ (3,2) -> 2,2 matrix
    # (3,2) @ (2,3) -> 3,3 matrix

    tensorA = torch.tensor([[1,2],[3,4],[5,6]])
    tensorB = torch.tensor([[7,10],[8,11],[9,12]])

    # print(torch.matmul(tensorA,tensorB)) this will error bc the sizes are wrong

    #to fix our issues we can manipulate one of the tensors using transpose
    #this switches the axis/dimn of a tensor

    print(torch.matmul(tensorA,tensorB.T))
    # size of 3,3 
    print(f"Original shapes: tensorA: {tensorA.shape}, tensorB: {tensorB.shape}")
    print(f"New shapes: tensorA: {tensorA.shape}, tensorB: {tensorB.T.shape}")
    print(f"Multiplying: {tensorA.shape} @ {tensorB.shape} <-- inner dimn must match")
    print("Output:\n")
    output = torch.matmul(tensorA,tensorB.T)
    print(output)
    print(output.shape)

    #tensor aggregation

    print("\n---Tensor Aggregation---\n")

    x = torch.arange(0,100,10)
    print(x)
    print(torch.min(x))
    print(x.max())
    print(x.sum())
    # print(x.mean()) type error it needs a float not a long
    print(x.type(torch.float32).mean())

    print()

    #to find the position of the min (index position where it is stored)
    print(x.argmin())
    print(x[x.argmin()])

    print()
    print(x.argmax())
    print(x[x.argmax()])

    # restacking,shaping,squeezing,and unsqueezing tensors

    # reshaping - reshapes an input tensor to a defined shape
    # view - rturn a view of an input tensor of a certain shape but keep the same memory as the og
    # stacking - combine multiple tnesors ontop or side by side to eachother
    # squeeze - removes all "1" dimension from a tensor
    # unqueeze - removes all "1" dimension from a tensor
    # permute - return a view of the input with dimensions 

    print("\n---Reshape---\n")
    x = torch.arange(0.,100.,20.)
    print(x)
    reshaped_x = x.reshape(1,5)
    print(reshaped_x)
    
    #change the view of x
    print("\n---View---\n")
    z = x.view(1,5)
    print(z)
    print(x)
    z[:,0] = 8
    print(z)
    print(x)

    print("\n---Stack---\n")

    x_stacked = torch.stack([x,x,x,x],dim=0)
    print(x_stacked)
    x_stacked = torch.stack([x,x,x,x],dim=1)
    print(x_stacked)

    print("\n---Squeeze---\n")

    x = torch.zeros(2, 1)
    print(x)
    print(x.size())
    
    y = torch.squeeze(x)
    print(y)
    print(y.size())

    y = torch.squeeze(x, 0)
    print(y)
    print(y.size())

    y = torch.squeeze(x, 1)
    print(y)
    print(y.size())

    print("\n---Unsqueeze---\n")

    x = torch.zeros(2)
    print(x)
    print(x.size())
    
    y = torch.unsqueeze(x)
    print(y)
    print(y.size())

    y = torch.unsqueeze(x, 0)
    print(y)
    print(y.size())

    y = torch.unsqueeze(x, 1)
    print(y)
    print(y.size())

    y = torch.unsqueeze(x, (1, 2, 3))
    print(y)
    print(y.size())