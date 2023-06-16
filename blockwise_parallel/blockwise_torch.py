import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from types import partial

def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.swish,
    "gelu_new": quick_gelu,
    "quick_gelu": quick_gelu,
}


MASK_VALUE = -1e10
Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024



def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 * (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i, j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, 0:sentinel] = cos

    return torch.tensor(out)


def rotate_every_two(tensor):
    rotate_half_tensor = torch.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), dim=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(1, 1, 2, 1)
    cos_pos = cos_pos[:, :, None, :].repeat(1, 1, 2, 1)
    return (torch * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class BlockwiseParallel(nn.Module):
    def __init__(self, hidden_size, num_heads, rotary_dim, intermediate_size, layer_norm_epsilon=1e-5,
                 activation_function="gelu", resid_pdrop=0.0, max_position_embeddings=1024, dtype=torch.float32,
                 casual=True, float32_logits=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rotary_dim = rotary_dim
        self.intermediate_size = intermediate_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.max_position_embeddings = max_position_embeddings
        self.dtype = dtype
        self.casual = casual
        self.float32_logits = float32_logits

        self.embed_dim = self.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        dense = partial(
            nn.Linear,
            self.embed_dim,
            bias=False,
            dtype=self.dtype
        )
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()
        self.ln_1 = nn.LayerNorm(self.hideen_size, eps=self.layer_norm_epsilon, elementwise_affine=True)

        self.ln_2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_epsilon, elementwise_affine=True)
        self.fc_in = nn.Linear(self.hidden_size, self.intermediate_size, dtype=self.dtype)
        self.fc_out = nn.Linear(self.intermediate_size, self.hidden_size, dtype=self.dtype)
        self.act = ACT2FN(self.activation_function)
        self.resid_pdrop = nn.Dropout(p=self.resid_pdrop)

        if self.rotary_dim is not None and self.rotary_dim > 0:
            pos_embd_dim = self.rotary_dim
        else:
            pos_embd_dim = self.embed_dim // self.num_heads
        self.embed_positions = create_sinusoidal_positions(self.max_position_embeddings, pos_embd_dim)


    def _split_heads(self, hidden_states):
        return hidden_states.view(hidden_states.shape[-2] + (self.num_heads, self.head_dim))
    
    def _merge_heads(self, hidden_states):
        return hidden_states.view(hidden_states.shape[:-2] + (self.embed_dim,))
    

    def attn_out_proj(self, attn_output, deterministic):
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_pdrop(attn_output)
        return attn_output
    
    def forward_qkv(self, hidden_states, position_ids, deterministic=True):
        hidden_states = self.ln_1(hidden_states)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)



        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rotary_dim is not None and self.rotary_dim > 0:
            sincos = self.embed_positions[position_ids].unsqueeze(1)
            q, k = apply_rotary_pos_emb(q, sincos), apply_rotary_pos_emb(k, sincos)

        return q, k, v

    def forward(self, hidden_states, position_ids, attention_mask=None, deterministic=True):
        q, k, v = self.forward_qkv(hidden_states, position_ids, deterministic)

        attn_output, attn_weights = self._attn(q, k, v, attention_mask, deterministic)
        attn_output = self.attn_out_proj(attn_output, deterministic)

        hidden_states = hidden_states + attn_output
        hidden_states = self.ln_2(hidden_states)

        ffn_output = self.fc_in(hidden_states)
        ffn_output = self.act(ffn_output)
        ffn_output = self.fc_out(ffn_output)
        ffn_output = self.resid_pdrop(ffn_output)

        hidden_states = hidden_states + ffn_output

        return hidden_states, attn_weights
    
    def _attn(self, q, k, v, attention_mask=None, deterministic=True):
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        if attention_mask is None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if not deterministic:
            attn_weights = self.resid_pdrop(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        return attn_output, attn_weights
    


