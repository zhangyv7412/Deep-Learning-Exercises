
import torch
import numpy as np

'''
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

x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2  # 生成一个布尔掩码
print(mask)   # tensor([False, False,  True,  True,  True])

# 用布尔掩码选出大于 2 的值
filtered_x = x[mask]
print(filtered_x)  # tensor([3, 4, 5])

# 用布尔掩码选出大于 2 的值,并赋值为0
x[mask]=0
print(x) # tensor([1, 2, 0, 0, 0])

tensor = torch.rand(3,4)
#以下是查看Tensor的形状,数据类型,设备的指令.
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#不改变元素顺序,把4*4矩阵变换为2*8
x = torch.randn(4,4)
y = x.reshape(2,8)
print(x)
print(y)
'''

#维度交换
x = torch.tensor([[1,2,3],[4,5,6]])
x_reshape = x.reshape(3,2)
x_transpose = x.permute(1,0) 
# permute的作用是交换指定维度.
# 此处Tensor的size输出是[2,3],分别是第0个位置和第1个位置.
# 这里的(1,0)相当于把原先的(0,1)换了位置.
# permute后括号里装的应该是用旧维度排列描述的新维度排列.
print('reshape:',x_reshape)
print('permute:',x_transpose)
