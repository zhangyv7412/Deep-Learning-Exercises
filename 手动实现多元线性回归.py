X = [[10,3],[20,3],[25,3],[28,2.5],[30,2],[35,2.5],[40,2.5]] #Feature
y = [60,85,100,120,140,145,163] #Label
#初始化参数
w = [0.0,0.0,0.0] #w0,w1,w2 线性方程中的参数
lr = 0.0001 #学习率
iteration = 10000 #迭代次数
#梯度下降算法
for i in range(iteration):
    #预测值
    y_pred = [w[0] + w[1] * x[0] + w[2] * x[1] for x in X]
    #损失函数
    loss = sum((y_pred[j] - y[j]) ** 2 for j in range(len(y))) / len(y)
    #计算梯度
    grad_w0 = 2 * sum(y_pred[j] - y[j] for j in range(len(y))) / len(y)
    grad_w1 = 2 * sum((y_pred[j] - y[j]) * X[j][0] for j in range(len(y))) / len(y)
    grad_w2 = 2 * sum((y_pred[j] - y[j]) * X[j][1] for j in range(len(y))) / len(y)
    #更新参数
    w[0] -= lr * grad_w0
    w[1] -= lr * grad_w1
    w[2] -= lr * grad_w2
    #打印损失
    if i % 100 == 0:
        print(f"Interration {i} : Loss = {loss}")
#输出最终参数
print(f"Final parameters: w0 = {w[0]}, w1 = {w[1]}, w2 = {w[2]}")


