
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

#只能用于二阶张量的转置方法
x = torch.rand(2,2)
x_transpose = x.t()
print(x)
print(x_transpose)

#扩展Tensor维度
x = torch.tensor([[1,2,3],[4,5,6]])
x_0 = x.unsqueeze(0)
print(x_0.shape,x_0)

#缩减Tensor维度
#squeeze方法会直接把大小为1的维度缩减掉,我们可以认为值为1的维度没有多余信息.
x = torch.ones((1,1,3))
print(x.shape,x)
y = x.squeeze(dim = 0)
print(y.shape, y)
z = x.squeeze()
print(z.shape,z)

#加减乘除,矩阵乘法.
a = torch.ones((2,3))
b = torch.ones(2,3)
print(a + b)
print(a - b)
print(a * b) #逐元素乘法
print(a / b) #逐元素除法
print(a @ b.t()) #矩阵乘法

#统计函数
t = torch.tensor([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

mean = t.mean()
print("mean:", mean)

mean = t.mean(dim=0)
print("mean on dim 0:", mean)

mean = t.mean(dim=0, keepdim=True)
print("keepdim:", mean)

#索引和切片
x = torch.tensor([[1,2,3],[4,5,6]])
print(x[0,1])
print(x[:,1])
print(x[1,:])
print(x[:,:2])
print(x[:,:])

#广播机制
t1 = torch.randn((3,2))
print(t1)
t2 = t1 + 1 
print(t2)

t1 = torch.ones((3,2))
t2 = torch.ones(2)

t3 = t1 + t2
print(t1)
print(t2)
print(t3)

x = torch.tensor(1.0, requires_grad=True) #指定需要计算梯度
y = torch.tensor(1.0, requires_grad=True) #指定需要计算梯度
v = 3*x+4*y
u = torch.square(v)
z = torch.log(u)

z.backward() #反向传播求梯度

#输出x,y在(1,1)点的梯度
print("x grad:", x.grad)
print("y grad:", y.grad)
