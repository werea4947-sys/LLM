from transformers import PretrainedConfig

#huggingface的类
class MokioMindConfig(PretrainedConfig):
    model_type = "myminimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

# 继承nn.module的类
class RMSNorm(nn.Module):
#__init__方法中定义了模型的结构和参数，forward方法中定义了模型的前向传播过程。
    def __init__(self,dim:int,eps:float=1e-5):
        super().init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
#_norm
    def _norm(self,x):
        return x*torch .rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)#rmsnorm的计算公式
    #其中x是输入张量，pow(2)表示对每个元素进行平方，mean(-1,keepdim=True)表示在最后一个维度上求平均值，并保持维度不变，eps是一个小常数，用于防止除以零的情况。最后将输入张量x乘以权重参数weight，并返回结果。
#forword方法中输入x，经过层归一化、线性变换、激活函数、dropout等操作，最后返回输出结果。
    def forward(self,x):
        output=self._norm(x)*self.weight.type_as(x)
        return output

def precompute_freqs_cis(dim:int,end: int = int(32 * 1024),rope_base: float = 1e6, rope_scaling: Optional[dict] = None):
    #初始化Rope频率
    freqs=1.0/(rope_base**(torch.arange(0,dim,2)[:(dim//2)].float()/dim))#arange(0,dim,2)为2i简化
    #[:(dim//2)]确保长度正好是维度的一半，因为每两个维度共享一个频率。

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        #推断的长度大于训练长度L，使用缩放
        if end/orig_max>1.0:
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
            
            #ramp为0时，缩放因子为1，表示不缩放；
            # ramp为1时，缩放因子为1/factor，表示最大缩放；
            # ramp在0和1之间时，缩放因子在1和factor之间线性变化。
            freqs=freqs*(1-ramp+ramp/factor)#根据缩放因子调整频率，得到最终的Rope频率。
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

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads # 本地注意力头数 假设8
        self.n_local_kv_heads = self.num_key_value_heads # 本地键值头数 2
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个键值头对应的注意力头数 4
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个注意力头的维度 假设512/8=64

        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False # 注意力头数乘以每个头的维度，得到查询向量的维度
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False # 键值头数乘以每个头的维度，得到键和值向量的维度
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False # 键值头数乘以每个头的维度，得到键和值向量的维度
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False # 注意力头数乘以每个头的维度，得到输出向量的维度
        )

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout) # 输出残差连接的dropout概率
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor], # 位置编码的余弦和正弦值
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # 之前的键值缓存，用于加速自回归推理
        use_cache=False, # 是否返回新的键值缓存，用于加速自回归推理
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码，用于屏蔽掉不需要关注的位置
    ):
        bsz, seq_len, _ = x.shape # 输入张量的形状，bsz是几句批量大小，seq_len是L序列长度，_是dim维度
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x) # 通过线性变换得到查询、键和值向量，
        #形状分别是[bsz, seq_len, num_attention_heads * head_dim]和[bsz, seq_len, num_key_value_heads * head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim) # 将查询向量重塑为[bsz, seq_len, 8, 64]的形状，以便后续的注意力计算
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) # 将键向量重塑为[bsz, seq_len, 2, 64]的形状，以便后续的注意力计算
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings # 位置编码的余弦和正弦值，形状是[seq_len, head_dim]，用于对查询和键进行旋转位置编码
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            #[bsz, 8n_local_heads, seq_len, 64head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        ##attention计算
        if (
            self.flash  # 如果支持flash attention，并且满足以下条件：
            # 训练模式，序列长度大于1，没有使用past_key_value，并且没有使用attention_mask或者attention_mask全为1，则直接使用torch的scaled_dot_product_attention函数进行计算。
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attention_mask=(
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1).expand(bsz,self.n_local_heads,seq_len,-1).bool()
            )
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attention_mask,
                dropout_p=self.dropout if self.training else 0.0,#训练模式下使用dropout，推理模式下不使用dropout
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)#计算点积注意力分数，xq的形状是[bsz, 8n_local_heads, seq_len, 64]，xk的形状是[bsz, 8n_local_heads, 64, seq_len]，通过矩阵乘法得到scores的形状是[bsz, 8n_local_heads, seq_len, seq_len]，然后除以sqrt(head_dim)进行缩放。
            scores=scores+torch.triu(torch.full((seq_len, seq_len),float("-inf"),device=scores.device),
                                     diagonal=1
            ).unsqueeze(0).unsqueeze(0)#使用torch.triu函数生成一个上三角矩阵，主对角线以上的元素为负无穷，主对角线及以下的元素为0。这个矩阵的形状是[seq_len, seq_len]，通过unsqueeze(0).unsqueeze(0)将其扩展为[1, 1, seq_len, seq_len]，然后与scores相加，实现了因果掩码的效果，即屏蔽掉未来位置的注意力分数。

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)# 对scores进行softmax归一化，得到注意力权重，保持与xq相同的数据类型。
            scores = self.attn_dropout(scores)
            output = scores @ xv#加权Value向量，得到注意力输出，形状是[bsz, 8n_local_heads, seq_len, 64]

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)#将注意力输出重塑为[bsz, seq_len, num_attention_heads * head_dim]的形状，以便后续的线性变换
        output = self.resid_dropout(self.o_proj(output))# 线性+残差
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]#激活函数

    def forward(self, x):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)#门控机制，先通过gate_proj得到一个中间表示，经过激活函数后与up_proj的输出相乘，得到gated张量。
        return self.dropout(self.down_proj(gated))

class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self,hidden_states,position_embeddings,past_key_value= None,use_cache=False,
        attention_mask= None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attention(self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        
        hidden_states = residual + hidden_states # 注意力的残差连接``
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vacab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer=nn.ModuleList(
            MokioMindBlock(i, config) for i in range(config.num_hidden_layers)
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #ROPE预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        def forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values:bool = None,
                **kwargs,
        ):
            batch_size, seq_len = input_ids.shape

            if hasattr(past_key_values,"layers"):
                past_key_values=None

            past_key_values=past_key_values or [None] * len(self.layer)

            start_pose=(
                past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
            )

            hidden_states = self.dropout(self.embed_tokens(input_ids))
            position_embeddings = (
                self.freqs_cos[start_pose : start_pose + seq_len], 
                self.freqs_sin[start_pose : start_pose + seq_len]
            )


            presents=[]

            for layer_idx, (layer, past_key_value) in enumerate(
                zip(self.layers, past_key_value)
            ):#循环k次，layer
                hidden_states, present= layer(
                    hidden_states,
                    position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=past_key_values is not None,
                    attention_mask=attention_mask,
                )
                
                presents.append(present)

            hiden_states = self.norm(hidden_states)

            return hiden_states, presents
    
class MokioMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        self.config = config

        super().__init__(config)

        self.model=MokioMindModel(config)
        
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        #权重共享，输出层与嵌入层的权重共享
        #可以减少模型参数数量，提升训练效率，同时也有助于模型更好地学习词汇之间的关系。
        self.model.embed_tokens.weight=self.lm_head.weight

        # #封装输出格式，方便与transformers库的接口兼容
        # self.OUT=CausalLMOutputWithPast()
    
    def forward(self,input_ids:Optional[torch.LongTensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values:Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep:Union[int,torch.Tensor]=0,
                **args,
        ):

        hidden_stastes,past_key_values=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )
        #logits_to_keep可以是一个整数，那就保留最后n个位置
        #生成时，只需关注最后的logits来预测下一个token
        slice_indings=(slice(-logits_to_keep,None) 
                       if isinstance(logits_to_keep,int) 
                       else logits_to_keep
        )
        logits=self.lm_head(hidden_stastes[:,slice_indings,:])

        # self.OUT.__setitem__("last_hidden_state",hidden_stastes)
        # self.OUT.__setitem__("logits",logits)
        # self.OUT.__setitem__("past_key_values",past_key_values)

        # return self.OUT

        output = CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_stastes,
        )
        return output