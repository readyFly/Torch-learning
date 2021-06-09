# Torch-learning

## 张量操作

### squezee & unsqueeze
* squezee 压缩维度
    ```
    torch.squeeze(input, dim=None, out=None) → Tensor
    ```
    除去输入张量input中数值为1的维度，并返回新的张量。如:输入张量的形状为（ A × 1 × B × C × 1 × D ） ，
    那么输出张量的形状为（ A × B × C × D ） 

    当通过dim参数指定维度时，维度压缩操作只会在指定的维度上进行。如果输入向量的形状为（ A × 1 × B ） ，
    squeeze(input, 0)会保持张量的维度不变，只有在执行squeeze(input, 1)时，输入张量的形状会被压缩至（ A × B ） 

    ps:如果一个张量只有1个维度，那么它不会受到上述方法的影响。

    输出的张量与原张量共享内存，如果改变其中的一个，另一个也会改变。

    Note:只能除去输入张量input中数值为1的维度。如果指定的维度不是1，那么不改变形状

    squeeze参数:

    input (Tensor) – 输入张量
    dim (int, optional) – 如果给定，则只会在给定维度压缩
    out (Tensor, optional) – 输出张量

* unsqueeze 新增维度
    ```
    torch.unsqueeze(input, dim) → Tensor
    ```
    新增加的这一个维度不会改变数据本身，只是为数据新增加了一个组别，这个组别是什么由我们自己定义。

    参数：
    input(tensor) -输入张量
    dim(int) -在给定维度上插入一维

    ```
    a=torch.rand(4,1,28,28)
    print(a.unsqueeze(0).shape)         #torch.Size([1, 4, 1, 28, 28]
    # 扩展一维，放于0维度

    b=torch.tensor([1.2, 2.3])          #torch.Size([2])
    print(b.unsqueeze(0)) #变成二维              #tensor([[1.2000, 2.3000]])   torch.Size([2, 1])
    print(b.unsqueeze(-1))
    #tensor([[1.2000],
    #        [2.3000]])

    x=torch.rand(3)
    y=torch.rand(4,32,14,14)
    print(x)
    print(x.unsqueeze(1))
    x=x.unsqueeze(1).unsqueeze(2).unsqueeze(0)      #[32]->[32,1]->[32,1,1]->[1,32,1,1]
    print(x.shape)                                  #torch.Size([1, 32, 1, 1])) 再进行扩展即可计算x+y
    ```

### view和reshape 调整tensor的shape,返回调整后的tensor 

```
a = torch.rand(4,1,28,28)

print(a.view(4,2,-1).shape)
# torch.Size([4, 2, 392])

print(a.reshape(4,-1).shape)
# torch.Size([4, 784])
```
相同点：都可以重新调整tensor的形状

不同点：view只能用于内存中连续存储的tensor。shape连续与否都能用。

如果对tensor做了transpose,permute等操作，则tensor在内存中会不连续，此时不能调用view函数。

此时先调用.contiguous()方法，使 tensor的元素在内存空间中连续，然后调用.view()

### expand维度扩展
torch.Tensor.expand(*sizes) → Tensor
将现有张量沿着值为1的维度扩展到新的维度。张量可以同时沿着任意一维或多维展开。
如果不想沿着一个特定的维度展开张量，可以设置它的参数值为-1。
参数：
sizes(torch.size or int....)--想要扩展的维度

```
x = torch.Tensor([3])
print(x.size())
# torch.Size([1])

print(x.expand(3,2))
# tensor([[3., 3.],
[3., 3.],
[3., 3.]])

a = torch.tensor([[[1,2,3],[4,5,6]]])
print(a.size()) 
# torch.Size([1, 2, 3])

print(a.expand(3,2,3)) #只能沿着1的维度扩展到新的维度

# tensor([[[1, 2, 3],
[4, 5, 6]],

[[1, 2, 3],
[4, 5, 6]],

[[1, 2, 3],
[4, 5, 6]]])
```

### transpose vs permute 

torch.transpose 只能交换两个维度 permute可以自由交换位置.

* For example:

    四个维度表示的[batch,channel,h,w] ，如果想把channel放到最后去，
    形成[batch,h,w,channel]，那么如果使用前面的维度交换，至少要交换两次（先13交换再12交换）。

    而使用permute可以直接指定维度新的所处位置，更加方便。

```
import torch
x = torch.rand(5,1,2,1)
print(x.size())

c = x.transpose(1,2) # 交换1和2维度
print(x.size())

b = x.permute(0,3,1,2) #可以直接变换 
print(b.size())  # torch.Size([5, 1, 1, 2])
```


### 转置

```
a=torch.rand(3, 4)
print(a.t().shape)   # .t操作指适用于矩阵
# torch.Size([4, 3])
```

### repeat维度重复：memory copied（增加了数据）
repeat会重新申请内存空间，repeat()参数表示各个维度指定的重复次数。

```
a=torch.rand(1,32,1,1)
print(a.repeat(4,32,1,1).shape)                 #torch.Size([4, 1024, 1, 1])
print(a.repeat(4,1,1,1,1).shape)                  #torch.Size([4, 32, 1, 1])
```

### 张量拼接 cat & stack
torch.cat(a_tuple, dim) 

tuple 是一个张量或者元组，在指定维度上进行拼接


torch.stack(a_tuple, dim) 

与cat不同的在于，cat只能在原有的某一维度上进行连接，stack可以创建一个新的维度，将原有维度在这个维度上进行顺序排列

比如说，有2个4x4的张量，用cat就只能把它们变成一个8x4或4x8的张量，用stack可以变成2x4x4.


### 缩小张量 narrow

```
torch.Tensor.narrow(dimension, start, length) → Tensor

```
返回一个经过缩小后的张量。
操作的维度由dimension指定。

缩小范围是从start开始到start+length。执行本方法的张量与返回的张量共享相同的底层内存。

参数：

dimension (int) – 要进行缩小的维度
start (int) – 开始维度索引
length (int) – 缩小持续的长度

for example:
```
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

m=x.narrow(0, 0, 2)#从行维度缩减，从0开始，缩小2的长度

print(m)
# tensor([[1., 2., 3.],
[4., 5., 6.]])

n=x.narrow(1, 1, 2)#从列维度缩减，从1开始，缩小2的长度

print(n)

# tensor([[2., 3.],
[5., 6.],
[8., 9.]])
```






### 设备转换 

```
x.to(device='cuda') #将数据转到GPU上

y.to(device='cpu') #将数据转到CPU上

```

### tensor与numpy的NDArray转换

pytorch的numpy()函数：tensor-->NDArray

pytorch的from_numpy()函数：NDArray-->tensor

```
import numpy as np

x = torch.rand(2,3)
ndarray = x.numpy() #x本身就是tensor,直接调用numpy()函数

ndarray1 = np.random.randn(2,3)
x1 = torch.from_numpy(ndarray1) #要使用torch.from_numpy()

```


### 张量创建和运算

pytorch中，张量是最基础的运算单位，与numpy中的NDArray类似，张量表示的是 一个多维矩阵。

不同的是，pytorch中的tensor可以运行在GPU上，基于numpy的计算 只能在CPU上。

> Tensor常用的基本数据类型：

    32位浮点型：torch.FloatTensor.  tensor的默认数据类型

    64位浮点型：torch.DoubleTensor

    64位整型：torch.LongTensor

    32位整型：torch.IntTensor

    16位整型：torch.ShortTensor

    此外，tensor还可以是byte或chart类型

> tensor初始化：
>> 1. rand():生成一个[0,1]简单的张量，例如2行3列的【0，1】的随机数tensor
>> 2. randn():初始化一个均值为0，方差为1的随机数tensor
>> 3. ones():初始化一个全为1的tensor
>> 4. zeros():初始化一个全为0的tensor
>> 5. eye():初始化一个主对角线为1，其余都为0的tensor（只能是二维矩阵）
>> 6. full(input,x):创建指定数值的张量,input是张量shape，x是填充数值
>> 7. arange():创建指定数值区间的张量
>> 8. linspace():创建等分-数列（等距）张量
>> 9. randint(begin,end,n):从给定的范围 [begin,end) 内生成n个随机整数
以上都可加_like(a),表示生成size和a一样的tensor
>> 10. normal():正态分布张量, 参数：（均值，标准差，张量尺寸）

```
x5 = torch.full((3,3),2)#矩阵尺寸3*3，填充数字为2

x6 = torch.arange(2,10,2) #数值区间[2,10),间隔2
# tensor([2, 4, 6, 8])

x7 = torch.linspace(2,10,6) #数值区间[2,10),创建元素个数6
# tensor([ 2.0000, 3.6000, 5.2000, 6.8000, 8.4000, 10.0000])

torch.normal(0,1,size=(4,))
# tensor([-0.2591, -0.3908, 2.2048, 1.4832])


```

> 基本运算

max():沿着行或列取最大值，参数dim=0表示沿着列，dim=1沿着行，返回value和idx. idx是取出的值在的每行或每列的位置

min():沿着行或列取最小值，参数dim=0表示沿着列，dim=1沿着行，返回value和idx

sum():沿着行或列求和, 再dim这个维度上，对tensor进行加和，返回结果会删掉这个维度

基本四则运算：+-*/ ，两个tensor必须大小一致，对应位置计算

add():加和，add_(),以_为结尾的会改变调用值本身

mm():矩阵乘法

> 索引 和 切片
```
a=torch.rand(4,3,28,28) #生成四维数据
print(a[0].shape)               #torch.Size([3, 28, 28])
print(a[0,0].shape)             #torch.Size([28, 28])
print(a[0,0,2,4])               #tensor(0.7309)

print(a[:2,:1].shape)           #torch.Size([2, 1, 28, 28])  等价于a[:2,:1,:,:].shape
print(a[:,:,::2,::2].shape)     #torch.Size([4, 3, 14, 14])
```

对于不规则的切片提取,可以使用torch.index_select, torch.take, torch.gather, torch.masked_select.
```
#抽取每个班级第0个学生，第5个学生，第9个学生的全部成绩
torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))

#抽取每个班级第0个学生，第5个学生，第9个学生的第1门课程，第3门课程，第6门课程成绩
q = torch.index_select(torch.index_select(scores,dim = 1,index = torch.tensor([0,5,9]))
                   ,dim=2,index = torch.tensor([1,3,6]))

#抽取第0个班级第0个学生的第0门课程，第2个班级的第4个学生的第1门课程，第3个班级的第9个学生第6门课程成绩
#take将输入看成一维数组，输出和index同形状
s = torch.take(scores,torch.tensor([0*10*7+0,2*10*7+4*7+1,3*10*7+9*7+6]))

#抽取分数大于等于80分的分数（布尔索引）
#结果是1维张量
g = torch.masked_select(scores,scores>=80)

```
如果要通过修改张量的部分元素值得到新的张量，可以使用torch.where,torch.index_fill 和 torch.masked_fill

```
#如果分数大于60分，赋值成1，否则赋值成0
ifpass = torch.where(scores>60,torch.tensor(1),torch.tensor(0))

#将每个班级第0个学生，第5个学生，第9个学生的全部成绩赋值成满分
torch.index_fill(scores,dim = 1,index = torch.tensor([0,5,9]),value = 100)

#将分数小于60分的分数赋值成60分
b = torch.masked_fill(scores,scores<60,60)

```
