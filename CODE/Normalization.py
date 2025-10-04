import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = torch.tensor([[2,1000],[3,2000],[2,500],[1,800],[4,3000]],dtype=torch.float , device=device)
labels = torch.tensor([[19],[31],[14],[15],[43]],dtype=torch.float, device=device)

#计算均值和标准差
mean = inputs.mean(dim= 0)
std = inputs.std(dim= 0)

#进行标准化
inputs_norm = (inputs - mean) / std

w = torch.ones(2, 1, requires_grad = True, device= device) 
b = torch.ones(1,requires_grad = True, device= device)

epoch = 1000
lr = 0.5


for i in range(epoch):
    outputs = inputs_norm @ w + b
    loss = torch.mean(torch.square(outputs - labels))
    print('loss',loss.item())
    loss.backward()
    print('w.grad', w.grad.tolist())
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr
    
    w.grad.zero_()
    b.grad.zero_()
print('训练后的w:',w)
print('训练后的b:',b)

# 对新采集的数据进行预测
new_input = torch.tensor([[3,2500]],dtype=torch.float,device=device)
# 对于新的数据进行预测时，同样要进行标准化
new_input = (new_input-mean)/std
# 预测
predict = new_input @ w + b
# 打印预测结果
print("Predict:",predict.tolist()[0][0])

