#ComputerScience/Python 

# 1. 数据类型
PyTorch里的数据类型，主要为：

**整数型**  
- torch.uint8
- torch.int32
- torch.int64  

torch.int64为默认的整数类型.

**浮点型**   
- torch.float16
- torch.bfloat16
- torch.float32
- torch.float64
 
torch.float32为默认的浮点数据类型.

**布尔型**   
- torch.bool

在PyTorch里使用最广泛的就是浮点型tensor.  
其中torch.float32称为全精度,torch.float16/torch.bfloat16称为半精度.  
一般情况下模型的训练是在全精度下进行的.  
如果采用混合精度训练的话，会在某些计算过程中采用半精度计算.  
混合精度计算会节省显存占用以及提升训练速度.  

在 PyTorch 中没有字符串类型, 因为 Tensor 设计之初就是用于数值计算, 不需要考虑字符串问题.  




> [!note] 布尔掩码
> 主要用于选择,过滤或修改元素等.  
>  一个例子:  
> ```python
> x = torch.tensor ([1,2,3])  
> mask = x > 2  
> print (mask)  # tensor ([False, False, True])
> ```

**注意:** 通过这种方式选择元素后,返回的是一个**一阶张量**,它只包含了被选中的值,原始张量的形状信息会丢失.  
