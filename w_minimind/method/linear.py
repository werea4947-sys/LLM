import torch
import torch.nn as nn
# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3])  # shape: (3,)

# t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# # 这里应用的w和b是随机的，真实训练里会在optimizer上更新
# output2 = layer(t2)             # shape: (1, 5)
# print(output2)

# t = torch.tensor([[ 1,  2,  3,  4,  5,  6],
#                   [ 7,  8,  9, 10, 11, 12]])
# t_view1 = t.view(3, 4)#view函数可以改变张量的形状，但不改变其数据。
# #这里将原来的2行6列的张量t重新组织成3行4列的张量t_view1。
# # 需要注意的是，view函数要求新的形状与原来的形状兼容，即新形状的元素总数必须与原形状的元素总数相同。在这个例子中，原来的张量有12个元素（2行*6列），新的张量也必须有12个元素（3行*4列）。因此，这个操作是合法的，并且t_view1将包含与t相同的数据，但以不同的形状组织。
# print(t_view1)
# t_view2 = t.view(4, 3)
# print(t_view2)

# #transpose函数可以交换张量的维度。这里将原来的2行6列的张量t转置成6行2列的张量t_transpose。
# t1=torch.Tensor([[1,2,3],[4,5,6]])#2x3
# t1=t1.transpose(0,1)#3x2交换行和列
# print(t1)

# x=torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])#3x3
# print(torch.triu(x))#返回输入矩阵的上三角部分，其他部分用0填充
# print(torch.triu(x, diagonal=1))#返回输入矩阵的上三角部分，diagonal=1表示从主对角线向上偏移一行开始保留元素，其他部分用0填充
# print(torch.triu(x, diagonal=-1))#返回输入矩阵的上三角部分，diagonal=-1表示从主对角线向下偏移一行开始保留元素，其他部分用0填充
# print(torch.tril(x))#返回输入矩阵的下三角部分，其他部分用0填充

x=torch.arange(0,12).view(1,12) # 1x12
Y=torch.reshape(x,(4,3)) # 4x3
print(Y)
z=torch.reshape(x,(3,-1)) # 3x4，-1表示自动计算维度大小以保持元素总数不变
print(z)