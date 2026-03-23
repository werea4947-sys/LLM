import torch

# #where函数可以根据条件选择元素，如果条件为True，则选择第一个输入张量中的元素，否则选择第二个输入张量中的元素。
# #condition是一个布尔张量，x和y是两个输入张量，result是一个新的张量，其中每个元素根据condition的值从x或y中选择。
# x=torch.tensor([1,2,3,4,5])
# y=torch.tensor([10,20,30,40,50])
# condition= x>3
# result=torch.where(condition,x,y)
# print(result)

# #arange函数可以创建一个一维的张量，包含从0到指定值（不包括该值）的整数。
# t=torch.arange(0,10,2)
# t1=torch.arange(5,0,-1)

# ## outer函数可以计算两个张量的外积，返回一个新的张量，其中每个元素是输入张量中对应元素的乘积。
# v1=torch.tensor([1,2,3])#一维张量，长度为3
# v2=torch.tensor([4,5,6])#一维张量，长度为3
# result=torch.outer(v1,v2)#返回一个大小为3x3的张量，其中每个元素是v1和v2中对应元素的乘积。
# print(result)

#cat函数可以将多个张量沿指定维度连接起来，返回一个新的张量。
# t1=torch.tensor([[[1,2,3],[4,5,6]],[[13,14,15],[16,17,18]]])#三维张量，大小为2x2x3
# t2=torch.tensor([[[7,8,9],[10,11,12]],[[19,20,21],[22,23,24]]])#三维张量，大小为2x2x3
# result=torch.cat((t1,t2),dim=0)#沿着第0维连接，返回一个大小为4x2x3的张量
# print(result,"在第0维连接，shape为4x2x3，意为着有4个2x3的子张量")

# result=torch.cat((t1,t2),dim=1)#沿着第1维连接，返回一个大小为2x4x3的张量
# print(result,"在第1维连接，shape为2x4x3，意为着有2个4x3的子张量")

# result=torch.cat((t1,t2),dim=-1)#沿着最后一个维度连接，返回一个大小为2x2x6的张量
# print(result,"在第2维连接，shape为2x2x6，意为着有2个2x6的子张量")

# t1=torch.tensor([1,2,3]) #一维数组，大小为3
# t2=t1.unsqueeze(0)#在第0维增加一个维度，1维数组变成了2维数组，大小为1x3
# print(t2)
# print(t2.shape)
# print(t1)
# print(t1.shape)


def precompute_freqs_cis(dim:int,end:int(32*1024),rope_base,rope_scaling:Optional[dict]):
    #初始化Rope频率
    freqs,attn_factor=(1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)),1.0)#arange(0,dim,2)为2i简化
    #[:(dim//2)]确保长度正好是维度的一半，因为每两个维度共享一个频率。

    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow=(
        rope_scaling["original_max_position_embeddings"],
        rope_scaling["factor"],
        rope_scaling["beta_fast"],
        rope_scaling["beta_slow"]
        )

        #推断的长度大于训练长度L，使用缩放
        if end>orig_max:
            #反解求i的公式，i=dim*log(orig_max/(b*2*pi))/(2*log(rope_base))，b为圈数，orig_max为训练时的最大位置编码长度，rope_base为Rope频率的底数。这个公式是根据Rope频率的定义和缩放关系推导出来的。
            inv_dim=lambda b:(dim*math.log(orig_max/(b*2*math.pi)))/(2*math.log(rope_base))

            ##划分高低维度
            #low：不需要缩放的高频，high：需要缩放的低频
            low,high=(max(math.floor(inv_dim(beta_fast)),0),#高频的最大i，向下取整
                      min(math.ceil(inv_dim(beta_slow)),dim//2-1))#低频的最小i，向上取整
            
            #计算缩放因子
            #low之前，ramp是0，high之后为1，中频为0-1。
            ramp=torch.clamp((torch.arange(dim//2,device=freqs.device).float()-low)
                             /max(high-low,0.001),0,1)
            #缩放因子=1+（factor-1）*ramp，factor是缩放的最大倍数，ramp是一个从0到1的线性函数，表示每个频率的缩放程度。
            #ramp为0时，缩放因子为1，表示不缩放；
            # ramp为1时，缩放因子为factor，表示最大缩放；
            # ramp在0和1之间时，缩放因子在1和factor之间线性变化。
            freqs=freqs*(1-ramp+ramp/factor)
        #根据end,生成位置索引t
        t=torch.arange(end,device=freqs.device).float()

        #计算外积，将t和频率freqs相乘，得到每个位置的旋转总角度
        freqs=torch.outer(t,freqs)
        freqs_cos=(torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=1)*attn_factor)
        freqs_sin=(torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=1)*attn_factor)

        return freqs_cos,freqs_sin

#编写ROPE位置编码函数
def apply_rotary_pos_emb(q,k,cos,sin,postion_ids=None,unsqueeze_dim=1):
    #[a,b]->[-b,a]，将偶数维和奇数维交错旋转
    def rotate_half(x):
        #x.shape[-1]//2表示维度的一半，
        # x[..., :x.shape[-1]//2]表示前半部分，
        # x[..., x.shape[-1]//2:]表示后半部分。通过切片操作将输入张量x分成两部分，并进行旋转。
        return torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],dim=-1)
    #x_rotated= x*cos + rotate_half(x)*sin，表示将输入张量x进行旋转，其中cos和sin分别是预先计算好的旋转角度的余弦和正弦值。rotate_half(x)表示将输入张量x进行半旋转，即将偶数维和奇数维交错旋转。
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
