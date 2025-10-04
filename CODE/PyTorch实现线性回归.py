import torch
#引入SummaryWriter库
from torch.utils.tensorboard import SummaryWriter

inputs = torch.rand(100,3) #生成训练集
weights = torch.tensor([[1.1],[2.2],[3.3]]) #初始化权重
bias = torch.tensor(4.4) #初始化偏置
#,加入一些误差作为噪音,生成Lable值
targets = inputs @ weights + bias + 0.1 * torch.randn(100,1) 

#创建一个SummaryWriter实例
writer = SummaryWriter()

#初始化参数
w = torch.rand((3,1),requires_grad=True)
b = torch.rand((1,),requires_grad=True)
epoch = 10000
lr = 0.003

#训练过程
for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print('loss:',loss.item())
    #记录标签名,Loss值,训练步数
    writer.add_scalar('loss/train',loss.item(),i)
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()


print('训练后的权重 w:',w)
print('训练后的偏置 b:',b)

