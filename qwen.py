import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from argparse import Namespace
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


def rope(config):
    base = config.rope_theta
    if "Qwen" in config.architectures[0]:
        rope_dim = config.hidden_size // config.num_attention_heads
    elif "DeepSeek" in config.architectures[0]:
        rope_dim = config.qk_rope_head_dim

    # 奇偶位分开编码
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).to(dtype=torch.float) / rope_dim)) #(rope_dim / 2)
    T = config.max_position_embeddings
    position_ids_expanded = torch.arange(0, T)
    freqs = torch.outer(position_ids_expanded, inv_freq).float() # (T, rope_dim / 2)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    # print(sin, cos); exit(-1)

    return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # print(cos, sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # print(q_embed, k_embed)
    # exit(-1)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))

    def forward(self, x):
        # import pdb; pdb.set_trace()
        ori_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        normed_x = x * rms_x
        return self.weight * normed_x.to(ori_dtype)


class SwishGELU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, bias=False):
        super(SwishGELU, self).__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)

    def forward(self, x):
        gate_x = self.gate_proj(x)
        x = self.up_proj(x)
        x =  F.silu(gate_x) * x
        x = self.down_proj(x)
        return x
    
class CausalGQA(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_groups, proj_bias=False):
        super(CausalGQA, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # print(hidden_dim, hidden_dim // (n_heads // n_groups))
        self.k_proj = nn.Linear(hidden_dim, hidden_dim // (n_heads // n_groups), bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim // (n_heads // n_groups), bias=True)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=proj_bias)

    def forward(self, hidden_state, cos=None, sin=None):
        # hidden_state.shape = (b, s, d)

        B, S, D = hidden_state.shape
        # print(B, S, D)
        # import pdb; pdb.set_trace()
        q = self.q_proj(hidden_state).reshape(B, S, self.n_heads, -1).transpose(1,2) #(b, n, s, d)
        # print(B, S, D)
        k, v = self.k_proj(hidden_state), self.v_proj(hidden_state)

        k = k.reshape(B, S, self.n_groups, -1).transpose(1,2)
        v = v.reshape(B, S, self.n_groups, -1).transpose(1,2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

        # print(q,k,v); exit(-1)
        k = k.repeat_interleave(dim=1, repeats=self.n_heads//self.n_groups)
        v = v.repeat_interleave(dim=1, repeats=self.n_heads//self.n_groups)
        # print(self.n_groups, self.n_heads)
        # print(q.shape, k.shape, v.shape, sin.shape, cos.shape)
        
        # Attention 计算
        # print(k, v); exit(-1)
        softmax_scale = 1 / (D // self.n_heads) ** 0.5
        # p = (q @ k.transpose(-1, -2))*softmax_scale
        # Attention mask
        # mask = torch.tril(torch.ones(S, S)).view(1, 1, S, S).to(q.device)
        # print(is_causal); exit(-1)
        o = F.scaled_dot_product_attention(q, k, v, scale=softmax_scale, is_causal=True)
        # p = torch.masked_fill(p, mask[:, :, :S, :S]==0, float('-inf'))
        # o = F.softmax(p, dim=-1, dtype=torch.float32).to(q.dtype) @ v # (b, n, s, d)
        # print(o); exit(-1)
        o = o.transpose(1, 2).contiguous().reshape(B, S, D)
        # print(k, v); exit(-1)
        # print(softmax_scale); exit(-1)
        out = self.o_proj(o)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, intermediate_dim, n_group):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.input_layernorm = RMSNorm(hidden_dim)
        self.post_attention_layernorm = RMSNorm(hidden_dim)
        self.self_attn = CausalGQA(hidden_dim=hidden_dim, n_heads=n_heads, n_groups=n_group)
        self.mlp = SwishGELU(hidden_dim=hidden_dim, intermediate_dim=intermediate_dim)

    def forward(self, hidden_state, cos=None, sin=None):
        # Residue
        # print(hidden_state.dtype, self.self_attn(self.input_layernorm(hidden_state), cos=cos, sin=sin).dtype)
        # print(hidden_state, self.input_layernorm(hidden_state))
        # exit(-1)
        # print(self.input_layernorm(hidden_state)); exit(-1)
        x1 = hidden_state+self.self_attn(self.input_layernorm(hidden_state), cos=cos, sin=sin)
        # print(x1); exit(-1)
        
        x2 = x1+self.mlp(self.post_attention_layernorm(x1))
        # print(x2); exit(-1)
        return x2

class Qwen2_5ForCasualLM(PreTrainedModel, GenerationMixin):
    config_class = Qwen2Config  # 绑定配置类
    base_model_prefix = "model"  # 与官方权重前缀一致
    supports_gradient_checkpointing = True  # 按需启用
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    # _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.model = QWen2_5(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @can_return_tuple
    # @auto_docstring
    def forward(self, input_ids, position_ids=None, **kwargs):
        # print(input_ids)
        # exit(-1)
        output, _ = self.model(input_ids, **kwargs)
        logits = self.lm_head(output[:, slice(-1, None, None), :])
        # print(logits)
        return CausalLMOutputWithPast(
            logits=logits
        )
    
    # @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, eos_token_id=None, temperature=1.0, top_k=50, attention_mask=None):
        idx = input_ids
        # Decode stage
        for _ in tqdm(range(max_new_tokens), desc="Generation Process"):
            input_idx = idx if idx.size(1) <= self.model.max_position_embeddings else idx[:, -self.model.max_position_embeddings:]
            logits = self(input_idx).logits

            logits =logits[:, -1] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits<v[:, -1]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)

            if eos_token_id is not None:
                if idx_next.item() in eos_token_id:
                    break
        return idx

class QWen2_5(PreTrainedModel):


    def __init__(self, config_dict):
        super(QWen2_5, self).__init__(config_dict)
        # config = Namespace(**config_dict)
        config = config_dict
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.max_position_embeddings = config.max_position_embeddings
        self.layers = nn.ModuleList(
            [TransformerBlock(config.hidden_size, n_heads=config.num_attention_heads, n_group=config.num_key_value_heads, intermediate_dim=config.intermediate_size) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # if config.tie_word_embeddings:
        #     self.embed_tokens.weight = self.lm_head.weight
        
        cos_cached, sin_cached = rope(config)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()
    #     elif isinstance(module, RMSNorm):
    #         module.weight.data.fill_(1.0)

    def forward(self, input_ids, **kwargs):
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids) # (B, S, D)
        # print(input_ids, x)
        # exit(-1)
        # cos, sin = 
        cos = self.cos_cached[:S, :].unsqueeze(0).expand(B, -1, -1)
        sin = self.sin_cached[:S, :].unsqueeze(0).expand(B, -1, -1)
        # print(cos, sin)
        # exit(-1)
        for layer in self.layers:
            x = layer(x, cos, sin)
            # print(x); exit(-1)
        # print(x); exit(-1)
        x = self.norm(x)
        # print(x)

        return x, None
    
    
    

if __name__ == "__main__":
    
# 将MyQwen注册到AutoModel系统
# AutoConfig.register("my_qwen", MyQwenConfig)
    # AutoModel.register(Qwen2Config, QWen2_5)
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    from transformers import AutoModel
    torch.manual_seed(42)


    # 加载官方模型的权重键名
    # official_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True, device_map='auto', torch_dtype='auto')
    # official_keys = set(official_model.state_dict().keys())
    # print({k:v.shape for k,v in official_model.state_dict().items()})

    # 获取您的模型权重键名
    # your_keys = set(QWen2_5().state_dict().keys())
    model = Qwen2_5ForCasualLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False
    )
    your_keys =model.state_dict().keys()
    # 找出不匹配的键
    # print("官方有但您没有的键:", official_keys - your_keys)  # 需要添加到您的模型
    # print("您有但官方没有的键:", your_keys - official_keys)  # 可能需要删除或重命名
    # exit(-1)

    # model = QWen2_5.from_pretrained(
    #     model_name,
    #     torch_dtype="auto",
    #     device_map="auto",
    #     trust_remote_code=False
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f'Loaded model on device {model.device} with type {model.dtype}')

    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        eos_token_id=[151645,151643],
        temperature=0.7,
        top_k=50
    )
    # print(generated_ids)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f'Prompt:{prompt}\nResponse:{response}')
