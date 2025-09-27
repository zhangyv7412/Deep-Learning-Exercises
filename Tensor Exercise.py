'''
import torch
import numpy as np

#1D tensor
t1 = torch.tensor([1,2,3])
print(t1)

#2D tensor
t2 = torch.tensor([[1,2,3],[4,5,6]])
print(t2)

#3D tensor
t3 = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(t3)

#Creat tensor from Numpy
arr = np.array([1,2,3])
t_np = torch.tensor(arr)
print(t_np)
'''
import torch
x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2  # 生成一个布尔掩码
print(mask)   # tensor([False, False,  True,  True,  True])

# 用布尔掩码选出大于 2 的值
filtered_x = x[mask]
print(filtered_x)  # tensor([3, 4, 5])


# 用布尔掩码选出大于 2 的值,并赋值为0
x[mask]=0
print(x) # tensor([1, 2, 0, 0, 0])