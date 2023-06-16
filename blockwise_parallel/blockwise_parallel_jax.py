# import jax
# import jax.numpy as jnp
# from jax import nn, lax
# from jax.experimental.stax import Dense

# class BlockwiseParallelTransformerAttention:
#     def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_seq_len = max_seq_len
#         self.block_size = block_size
#         self.dim_per_head = hidden_size // num_heads

#         self.query_chunk_size = max_seq_len // block_size
#         self.key_value_chunk_size = max_seq_len // block_size
#         self.num_query_chunks = (max_seq_len + self.query_chunk_size - 1) // self.query_chunk_size
#         self.num_key_value_chunks = (max_seq_len + self.key_value_chunk_size - 1) // self.key_value_chunk_size

#         self.query_position_ids = jnp.arange(max_seq_len)
#         self.key_value_position_ids = jnp.arange(max_seq_len)

#         self.query_blocks = Dense(hidden_size, name='query')
#         self.key_blocks = Dense(hidden_size, name='key')
#         self.value_blocks = Dense(hidden_size, name='value')
#         self.feedforward = Dense(hidden_size, name='feedforward')

#     def _chunk_bias_fn(self, query_chunk_idx, key_chunk_idx):
#         start = key_chunk_idx * self.key_value_chunk_size
#         end = (key_chunk_idx + 1) * self.key_value_chunk_size
#         bias_chunk = jnp.zeros((self.num_heads, self.query_chunk_size, self.key_value_chunk_size))
#         bias_chunk = lax.dynamic_update_slice(bias_chunk, jnp.ones((self.num_heads, self.query_chunk_size, end - start)), (slice(None), slice(None), slice(start, end)))
#         bias_chunk = jnp.expand_dims(bias_chunk, axis=0)
#         bias_chunk = jnp.tile(bias_chunk, (query_chunk_idx.shape[0], 1, 1, 1))
#         return bias_chunk

#     def _query_block(self, input_chunk, query_chunk_idx):
#         query_chunk = self.query_blocks(input_chunk)
#         query_chunk = query_chunk / jnp.sqrt(query_chunk.shape[-1])
#         return query_chunk

#     def _key_value_blocks(self, carry, args):
#         kv_chunk, key_chunk_idx, kv_position_ids_chunk = args
#         query_chunk, query_chunk_idx = carry
#         key_chunk = self.key_blocks(kv_chunk)
#         value_chunk = self.value_blocks(kv_chunk)
#         attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)
#         bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
#         bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
#         attn_weights = attn_weights + bias_chunk
#         max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
#         exp_weights = jnp.exp(attn_weights - max_score)
#         exp_values = jnp.einsum('bqhv,bvhf->bqhf', exp_weights, value_chunk)
#         numerator = jax.lax.dynamic_update_slice(query_chunk, exp_values, (slice(None), key_chunk_idx, slice(None), slice(None)))
#         denominator = jax.lax.dynamic_update_slice(query_chunk, exp_weights.sum(axis=-1, keepdims=True), (slice(None), key_chunk_idx, slice(None), slice(None)))
#         return (numerator, denominator), None
    
#     def __call__(self, x, deterministic=True):
#         batch_size, seq_len, input_size = x.shape
#         assert input_size == self.input_size, f"Input size must be {self.input_size} but got {input_size}"

#         query_chunks = x.reshape(batch_size, self.num_query_chunks, self.query_chunk_size, input_size)
#         query_chunks = self.query_blocks(query_chunks)

#         query_chunks = query_chunks / jnp.sqrt(query_chunks.shape[-1])
#         query_position_ids = jnp.tile(self.query_position_ids, (batch_size, 1))
#         query_position_ids = query_position_ids.reshape(batch_size, self.num_query_chunks, self.query_chunk_size)
#         query_position_ids = jax.lax.dynamic_slide(query_position_ids, (0, 0, 0), (batch_size, self.num_query_chunks, self.query_chunk_size - 1))
#         query_position_ids = jnp.concatenate([query_position_ids, jnp.ones((batch_size, self.num_query_chunks, 1)) * (self.max_seq_len - 1)], axis=-1)
#         query_position_ids = query_position_ids.astype(jnp.int32)

#         key_value_chunks = x.reshape(batch_size, self.num_key_value_chinks, self.key_value_chunk_size, input_size)
#         key_value_chunks = jax.lax.stop_gradient(key_value_chunks) if deterministic else key_value_chunks
#         key_value_position_ids = jnp.tile(self.key_value_position_ids, (batch_size, 1))
#         key_value_position_ids = key_value_position_ids.reshape(batch_size, self.num_value_chunks, self.key_value_chunk_size)
#         key_value_position_ids = jax.lax.dynamic_slice(key_value_position_ids, (0, 0, 0), (batch_size, self.num_key_value_chunks, self.key_value_chunk_size - 1))
#         key_value_position_ids = jnp.concatenate([key_value_position_ids, jnp.ones((batch_size, self.num_key_value_chunks, 1)) * (self.max_seq_len - 1)], axis=-1)
#         key_value_position_ids = key_value_position_ids.astype(jnp.int32)

#         query_blocks = jax.lax.map(self._query_block, query_chunks, jnp.arange(self.num_query_chunks))
#         query_blocks = query_blocks.reshape(batch_size, self.num_query_chunks, self.num_heads, self.query_chunk_size, self.dim_per_head)
#         query_blocks = jnp.moveaxis(query_blocks, 2, 3)


#         key_value_blocks = key_value_chunks.reshape(batch_size, self.num_key_value_chunks, self.num_heads, self.key_value_chunk_size, self.dim_per_head)
#         key_value_blocks = jnp.moveaxis(key_value_blocks, 2, 3)

#         carry = (query_blocks, None)
#         key_value_blocks = jax.lax.scan(self._key_value_blocks, carry, (key_value_blocks, jnp.arange(self.num_key_value_chunks), key_value_position_ids))[0][0]

#         key_value_blocks = jnp.moveaxis(key_value_blocks, 2, 3)
#         key_value_blocks = key_value_blocks.reshape(batch_size, self.num_key_value_chunks, self.key_value_chunk_size, self.hidden_size)

#         output = jax.lax.map(lambda x: self.feedforward(x.reshape(-1, self.hidden_size)), key_value_blocks)
#         output = output.reshape(batch_size, seq_len, self.hidden_size)

#         return output

    
    
    
    
    

#==================================== v2


# import jax
# import jax.numpy as jnp
# from jax.experimental import stax

# class BlockwiseParallelTransformerAttention(nn.Module):
#     def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
#         super(BlockwiseParallelTransformerAttention, self).__init__()
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_seq_len = max_seq_len
#         self.block_size = block_size
        
#         self.query_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.key_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.value_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.feedforward = nn.Sequential(
#             stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal()),
#             nn.ReLU(),
#             stax.Dense(num_heads * hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         )
#         self.layer_norm1 = nn.LayerNorm(input_size)
#         self.layer_norm2 = nn.LayerNorm(num_heads * hidden_size)
        
#     def forward(self, x):
#         batch_size, seq_len, input_size = x.shape
#         num_blocks = seq_len // self.block_size
#         query_blocks = x[:, :num_blocks*self.block_size, :].reshape(batch_size, num_blocks, self.block_size, input_size)
#         key_value_blocks = x[:, :num_blocks*self.block_size, :].reshape(batch_size, num_blocks, self.block_size, input_size)
        
#         for i in range(self.num_layers):
#             query = self.query_blocks(query_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
#             key = self.key_blocks(key_value_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
#             value = self.value_blocks(key_value_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
            
#             query = query.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
#             key = key.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
#             value = value.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
            
#             attention_scores = jnp.matmul(query, key.transpose((0, 1, 2, 4, 3))) / jnp.sqrt(jnp.array(self.hidden_size, dtype=jnp.float32))
#             attention_weights = nn.functional.softmax(attention_scores, dim=-1)
#             attention_output = jnp.matmul(attention_weights, value)
#             attention_output = attention_output.transpose((0, 2, 3, 1, 4)).reshape(batch_size*num_blocks, self.block_size, self.num_heads*self.hidden_size)
#             attention_output = self.feedforward(attention_output)
#             attention_output = attention_output.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 2, 1, 3, 4)).reshape(batch_size, seq_len, self.num_heads*self.hidden_size)
#             attention_output = self.layer_norm1(query_blocks + attention_output)
#             attention_output = self.layer_norm2(attention_output)
            
#         return attention_output









    
    
    
    
    
    
    
    
    
    
    
    # def __call__(self, x, deterministic=True):
    #     batch_size, seq_len, input_size = x.shape
    #     assert input_size == self.input_size, f'Input size must be {self.input_size}, but got {input_size}'

    #     query_chunks = x.reshape(batch_size, self.num_query_chunks, self.query_chunk_size, input_size)
    #     query_chunks = self.query_blocks(query_chunks)
    #     query_chunks = query_chunks / jnp.sqrt(query_chunks.shape[-1])

    #     kv_chunks = x.reshape(batch_size, self.num_key_value_chunks, self.key_value_chunk_size, input_size)
    #     kv_chunks = self.key_blocks(kv_chunks), self.value_blocks(kv_chunks)

    #     init_carry = (jnp.zeros((batch_size, self.query_chunk_size, self.num_heads, self.dim_per_head)),
    #                   jnp.zeros((batch_size, self.query_chunk_size, self.num_heads, self.dim_per_head)),
    #                   (-jnp.inf) * jnp.ones((batch_size, self.query_chunk_size, self.num_heads, 1)))
            
    #     def attention_block(carry, args):
    #         query_chunk, query_chunk_idx = carry
    #         kv_chunk, key_chunk_idx, kv_position_ids_chunk = args

    #         key_chunk, value_chunk = kv_chunk
    #         attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)
    #         bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
    #         bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
    #         attn_weights = attn_weights + bias_chunk
    #         max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    #         exp_weights = jnp.exp(attn_weights - max_score)
    #         exp_values = jnp.einsum('bqhv,bvhf->bqhf', exp_weights, value_chunk)
    #         numerator = jax.lax.dynamic_update_slice(query_chunk, exp_values, (slice(None), query_chunk_idx, slice(None), slice(None)))
    #         denominator = jax.lax.dynamic_update_slice(query_chunk, exp_weights.sum(axis=-1, keepdims=True), (slice(None), query_chunk_idx, slice(None), slice(None)))
    #         return (numerator, denominator), None

    #     def combine_blocks(carry, args):
    #         query_chunk, query_chunk_idx = carry
    #         numerator, denominator = args
    #         numerator = jnp.concatenate([query_chunk, numerator], axis=2)
    #         denominator = jnp.concatenate([jnp.ones_like(query_chunk), denominator], axis=2)
    #         attn_output = jnp.sum(numerator / denominator, axis=2)
    #         attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
    #         attn_output = attn_output + x
    #         return (attn_output, query_chunk_idx + 1), None

    #     def feedforward_block(x):
    #         hidden = self.feedforward(x)
    #         hidden = nn.gelu(hidden)
    #         return hidden + x

    #     for layer_idx in range(self.num_layers):
    #         query_chunk_idx = 0
    #         carry = (query_chunks[:, query_chunk_idx], query_chunk_idx)
    #         for key_chunk_idx in range(self.num_key_value_chunks):
    #             kv_chunk = kv_chunks[:, key_chunk_idx]
    #             kv_position_ids_chunk = self.key_value_position_ids[key_chunk_idx * self.key_value_chunk_size:(key_chunk_idx + 1) * self.key_value_chunk_size]
    #             carry, _ = BlockParallel(self.num_heads)(attention_block, carry, (kv_chunk, key_chunk_idx, kv_position_ids_chunk))
    #         attn_output, _ = BlockParallel()(combine_blocks, carry, None)
    #         x = attn_output
    #         x = BlockParallel()(feedforward_block, x)
    #     return x
            
        
        
        
        
        
        
        
        
        
    #     # for key_chunk_idx in range(self.num_key_value_chunks):
    #     #     for key_chunk_idx in range(self.num_key_value_chunks):
    #     #         key_value_chunk = kv_chunks[:, key_chunk_idx]
    #     #         key_value_position_ids_chunk = self.key_value_position_ids[key_chunk_idx * self.key_value_chunk_size:(key_chunk_idx + 1) * self.key_value_chunk_size]
    #     #         carry, _ = lax.scan(self._key_value_blocks, carry, (key_value_chunk, key_chunk_idx, key_value_position_ids_chunk))

    #     #     numerator, denominator, bias = carry
    #     #     attn_weights = numerator / denominator
    #     #     attn_weights = jax.lax.dynamic_update_slice(attn_weights, bias, (slice(None), slice(None), slice(None), 0))
    #     #     attn_weights = nn.softmax(attn_weights, axis=-2)
    #     #     attn_weights = jax.lax.dynamic_update_slice(attn_weights, jnp.zeros_like(bias), (slice(None), slice(None), slice(None), 0))

    #     #     value_chunk = jnp.einsum('bqhv,bvhf->bqhf', attn_weights, kv_chunks)
    #     #     value_chunk = value_chunk.reshape(batch_size, self.num_heads * self.query_chunk_size, self.dim_per_head)
    #     #     value_chunk = self.feedforward(value_chunk)
    #     #     value_chunk = value_chunk.reshape(batch_size, self.num_heads, self.query_chunk_size, self.dim_per_head)
    #     #     value_chunk = jnp.moveaxis(value_chunk, 1, 2)
    #     #     if query_chunk_idx == 0:
    #     #         output = value_chunk
    #     #     else:
    #     #         output = jnp.concatenate([output, value_chunk], axis=2)

    #     # output = output.reshape(batch_size, seq_len, self.hidden_size)
    #     # return output

    #     # # def _key_value_blocks(cell, carry, args):
    #     # #     kv_chunk, key_chunk_idx, kv_position_ids_chunk = args
    #     # #     query_chunk, query_chunk_idx = carry
    #     # #     key_chunk = self.key_blocks(kv_chunk)
    #     # #     value_chunk = self.value_blocks(kv_chunk)
    #     # #     attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)
    #     # #     bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
    #     # #     bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
    #     # #     attn_weights = attn_weights + bias_chunk
    #     # #     max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
    #     # #     exp_weights = jnp.exp(attn_weights - max_score)
    #     # #     exp_values = jnp.einsum('bqhv,bvhf->bqhf', exp_weights, value_chunk)
    #     # #     numerator = jax.lax.dynamic_update_slice(query_chunk, exp_values, (slice(None), key_chunk_idx, slice(None), slice(None)))
    #     # #     denominator = jax.lax.dynamic_update_slice(query_chunk, exp_weights.sum(axis=-1, keepdims=True), (slice(None), key_chunk_idx, slice(None), slice(None)))
    #     # #     return (numerator, denominator), None
        
    #     # # for query_chunk_idx in range(self.num_query_chunks):
    #     # #     query_chunk = self._query_block(query_chunks[:, query_chunk_idx], query_chunk_idx)
    #     # #     for key_value_chunk_idx in range(self.num_key_value_chunks):
    #     # #         kv_chunk = kv_chunks[:, key_value_chunk_idx, :, :]
    #     # #         init_carry = (query_chunk, query_chunk_idx)
    #     # #         (numerator, denominator), _ = lax.scan(_key_value_blocks, init_carry, (kv_chunk, key_value_chunk_idx))
    #     # #     attention_output_chunk = numerator / denominator 
    #     # #     attention_output_chunk = self.feedforward(attention_output_chunk)
    #     # #     query_chunk = query_chunks[:, query_chunk_idx]
    #     # #     attention_output_chunk = attention_output_chunk + query_chunk
    #     # #     attention_output_chunk = nn.LayerNorm(attention_output_chunk)
    #     # #     query_chunks = jax.lax.dynamic_update_slice(query_chunks, attention_output_chunk, (slice(None), query_chunk_idx, slice(None), slice(None)))
        
    #     # # attention_output = query_chunks.reshape(batch_size, seq_len, self.hidden_size)
    #     # # return attention_output
        

        
    # def BlockParallel(num_blocks=None):
    #     def decorator(f):
    #         def wrapper(*args, **kwargs):
    #             if num_blocks is None:
    #                 num_blocks = jax.local_device_count()
    #             block_size = args[0].shape[0] // num_blocks
    #             blocks = [jax.lax.dynamic_slice_in_dim(args[0], i * block_size, block_size, axis=0) for i in range(num_blocks)]
    #             args = [(block,) + args[1:] for block in blocks]
    #             outputs = jax.pmap(f)(*args, **kwargs)
    #             return jnp.concatenate(outputs, axis=0)
    #         return wrapper
#     #     return decorator
#     import jax
# import jax.numpy as jnp
# from jax.experimental import stax

# class BlockwiseParallelTransformerAttention(nn.Module):
#     def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
#         super(BlockwiseParallelTransformerAttention, self).__init__()
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_seq_len = max_seq_len
#         self.block_size = block_size
        
#         self.query_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.key_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.value_blocks = stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         self.feedforward = nn.Sequential(
#             stax.Dense(hidden_size, W_init=jax.nn.initializers.glorot_normal()),
#             nn.ReLU(),
#             stax.Dense(num_heads * hidden_size, W_init=jax.nn.initializers.glorot_normal())
#         )
#         self.layer_norm1 = nn.LayerNorm(input_size)
#         self.layer_norm2 = nn.LayerNorm(num_heads * hidden_size)
        
#     def forward(self, x):
#         batch_size, seq_len, input_size = x.shape
#         num_blocks = seq_len // self.block_size
#         query_blocks = x[:, :num_blocks*self.block_size, :].reshape(batch_size, num_blocks, self.block_size, input_size)
#         key_value_blocks = x[:, :num_blocks*self.block_size, :].reshape(batch_size, num_blocks, self.block_size, input_size)
        
#         for i in range(self.num_layers):
#             query = self.query_blocks(query_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
#             key = self.key_blocks(key_value_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
#             value = self.value_blocks(key_value_blocks.reshape(batch_size*num_blocks, self.block_size, input_size))
            
#             query = query.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
#             key = key.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
#             value = value.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 3, 1, 2, 4))
            
#             attention_scores = jnp.matmul(query, key.transpose((0, 1, 2, 4, 3))) / jnp.sqrt(jnp.array(self.hidden_size, dtype=jnp.float32))
#             attention_weights = nn.functional.softmax(attention_scores, dim=-1)
#             attention_output = jnp.matmul(attention_weights, value)
#             attention_output = attention_output.transpose((0, 2, 3, 1, 4)).reshape(batch_size*num_blocks, self.block_size, self.num_heads*self.hidden_size)
#             attention_output = self.feedforward(attention_output)
#             attention_output = attention_output.reshape(batch_size, num_blocks, self.block_size, self.num_heads, self.hidden_size).transpose((0, 2, 1, 3, 4)).reshape(batch_size, seq_len, self.num_heads*self.hidden_size)
#             attention_output = self.layer_norm1(query_blocks + attention_output)
#             attention_output = self.layer_norm2(attention_output)
            
#         return attention_output


















#==================================== v3
import functools
import json
import math
from functools import partial
from typing import Callable, NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax.linen import combine_masks, make_causal_mask
from jax import lax
from jax import numpy as jnp


def quick_gelu(x):
    return x * jax.nn.sigmoid(1.702 * x)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'dots_saveable': jax.checkpoint_policies.dots_saveable,
        'dots_with_no_batch_dims_saveable': jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    }[name]

MASK_VALUE = -1e10

Q_CHUNK_SIZE = 1024
K_CHUNK_SIZE = 1024

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)


def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class _AttentionBlock(nn.Module):
    hidden_size: int
    num_heads: int
    rotary_dim: Optional[int]
    intermediate_size: int
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    resid_pdrop: float = 0.0
    max_position_embeddings: int = 1024
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    float32_logits: bool = False

    def setup(self):
        self.embed_dim = self.hidden_size
        self.head_dim = self.embed_dim // self.num_heads
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in',
                distribution='normal',
            )
        )
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        self.out_proj = dense()
        self.ln_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)

        self.ln_2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)
        self.fc_in = nn.Dense(self.intermediate_size,
                            dtype=self.dtype,
                            kernel_init=jax.nn.initializers.variance_scaling(
                            scale=1.0, mode='fan_in',
                            distribution='normal',
            )
        )
        self.fc_out = nn.Dense(self.embed_dim,
                            dtype=self.dtype,
                            kernel_init=jax.nn.initializers.variance_scaling(
                            scale=1.0, mode='fan_in',
                            distribution='normal',
            )
        )
        self.act = ACT2FN[self.activation_function]
        self.resid_dropout = nn.Dropout(rate=self.resid_pdrop)

        if self.rotary_dim is not None and self.rotary_dim > 0:
            pos_embd_dim = self.rotary_dim
        else:
            pos_embd_dim = self.embed_dim // self.num_heads
        self.embed_positions = create_sinusoidal_positions(self.max_position_embeddings, pos_embd_dim)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def attn_out_proj(self, attn_output, deterministic):
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        return attn_output

    def forward_qkv(
        self,
        hidden_states,
        position_ids,
        deterministic: bool = True,
    ):
        hidden_states = self.ln_1(hidden_states)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        sincos = jnp.take(self.embed_positions, position_ids, axis=0)
        sincos = jnp.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None and self.rotary_dim > 0:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)

        if self.float32_logits:
            query = query.astype(jnp.float32)
            key = key.astype(jnp.float32)

        return query, key, value

    def forward_ffn(
        self,
        hidden_states,
        deterministic: bool = True,
    ):
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.resid_dropout(hidden_states, deterministic=deterministic)

        return hidden_states


class AttentionBlock(nn.Module):
    q_chunk_size: int
    k_chunk_size: int
    hidden_size: int
    num_heads: int
    rotary_dim: Optional[int]
    intermediate_size: int
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    max_position_embeddings: int = 1024
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    policy: str = 'nothing_saveable'
    prevent_cse: bool = False
    float32_logits: bool = False

    def setup(self):
        self.attn = _AttentionBlock(
            self.hidden_size,
            self.num_heads,
            self.rotary_dim,
            self.intermediate_size,
            self.layer_norm_epsilon,
            self.activation_function,
            self.resid_pdrop,
            self.max_position_embeddings,
            self.dtype,
            self.causal,
            self.float32_logits,
        )

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
    ):
        query, key, value = self.attn.forward_qkv(hidden_states, position_ids)
        query = query / jnp.sqrt(query.shape[-1])

        dropout_rng = None
        if not deterministic and self.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, -1e9).astype(self.dtype),
        )

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            query, key, value = self.attn.forward_qkv(hidden_states, position_ids)
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)
            # use standard dot product attention since query length is 1
            attn_weights = nn.dot_product_attention_weights(
                query,
                key,
                bias=attention_bias,
                dropout_rng=dropout_rng,
                dropout_rate=self.config.attn_pdrop,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=None,
            )
            attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
            attn_output = self.attn.attn_out_proj(attn_output, deterministic=deterministic)
            ffn_output = self.attn.forward_ffn(hidden_states + attn_output, deterministic=deterministic)
            outputs = attn_output + ffn_output + hidden_states
        else:
            attn_output = blockwise_compute_attn(
                query,
                key,
                value,
                bias=attention_bias,
                deterministic=not deterministic,
                dropout_rng=dropout_rng,
                attn_pdrop=self.attn_pdrop,
                causal_mask=self.causal,
                query_chunk_size=self.q_chunk_size,
                key_chunk_size=self.k_chunk_size,
                dtype=self.dtype,
                policy=self.policy,
                precision=None,
                prevent_cse=self.prevent_cse,
            )
            attn_output = self.attn.attn_out_proj(attn_output, deterministic=deterministic)
            ffn_output = blockwise_compute_ffn(
                self.attn,
                hidden_states + attn_output,
                chunk_size=self.q_chunk_size,
                deterministic=deterministic,
                policy=self.policy,
                prevent_cse=self.prevent_cse,
            )
            outputs = ffn_output + hidden_states + attn_output
        return outputs


def _chunk_attention_bias(query_chunk_size, key_chunk_size,
            bias, deterministic, attn_dropout, attn_pdrop, causal_mask,
            query_chunk_idx, key_chunk_idx):
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1))
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(*bias.shape[:2], min(bias.shape[-2], query_chunk_size), min(bias.shape[-1], key_chunk_size)),
        )

    if causal_mask:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        offset = query_offset - key_offset
        query_idx += offset
        causal_mask_value = (query_idx < key_idx) * MASK_VALUE
        chunk_bias += causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias -= attn_dropout_slice * 1e6
    return chunk_bias

class Carry(NamedTuple):
    numerator: jax.Array
    denominator: jax.Array
    max_so_far: jax.Array

def blockwise_compute_attn(query, key, value,
        bias=None,
        deterministic=False,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal_mask=True,
        query_chunk_size=None,
        key_chunk_size=None,
        dtype=jnp.float32,
        policy='nothing_saveable',
        precision=lax.Precision.HIGHEST,
        prevent_cse=False,):
    q_len = query.shape[1]
    kv_len = key.shape[1]
    query = rearrange(query, 'b (n c) h q -> b n c h q', c=query_chunk_size)
    key, value = map(lambda t: rearrange(t, 'b (n c) h v -> b n c h v', c=key_chunk_size), (key, value))
    query, key, value = map(lambda t: rearrange(t, 'b n c h d -> n b c h d'), (query, key, value))
    num_q, batch, _, num_heads, dim_per_head = query.shape
    num_kv, _, _, _, _ = key.shape

    for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
        assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size,
        bias, deterministic, attn_dropout, attn_pdrop, causal_mask)

    def _query_chunk_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse,
                           policy=get_gradient_checkpoint_policy(policy))
        def summarize_chunk(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum(
                'bqhv,bvhf->bqhf', exp_weights, value_chunk, precision=precision
            )
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return Carry(numerator, denominator, max_score), None

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            summarize_chunk, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(
        lambda _, x: ((), _query_chunk_attention(x)),
        (), xs=(query, jnp.arange(0, num_q))
    )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res

def blockwise_compute_ffn(cell, inputs, chunk_size, deterministic, policy, prevent_cse):
    inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=chunk_size)
    inputs = rearrange(inputs, 'b n c d -> n b c d')
    num_q, _, _, _ = inputs.shape
    def ffn(cell, _, hidden_states):
        outputs = cell.forward_ffn(hidden_states, deterministic=deterministic)
        return _, outputs
    ffn_remat = nn.remat(
        ffn,
        variables="params",
        rngs={"params" : False},
        prevent_cse=prevent_cse,
        policy=get_gradient_checkpoint_policy(policy),
    )
    _, res = nn.scan(
        ffn_remat,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
        length=num_q,
    )(cell, None, inputs)
    res = rearrange(res, 'n b c d -> b (n c) d')
    return res

class Blockwise_LM_Head(nn.Module):
    vocab_size: int
    chunk_size: int
    policy: str = 'nothing_saveable'
    dtype: jnp.dtype = jnp.float32
    prevent_cse: bool = False

    def setup(self):
        self.lm_head = nn.Dense(
            self.vocab_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in',
                distribution='normal',
            )
        )

    def __call__(self, inputs):
        inputs = rearrange(inputs, 'b (n c) d -> b n c d', c=self.chunk_size)
        inputs = rearrange(inputs, 'b n c d -> n b c d')
        num_q, _, _, _ = inputs.shape
        def lm_head(cell, _, hidden_states):
            outputs = cell(hidden_states)
            return _, outputs
        lm_head_remat = nn.remat(
            lm_head,
            variables="params",
            rngs={"params" : False},
            prevent_cse=self.prevent_cse,
            policy=get_gradient_checkpoint_policy(self.policy),
        )
        _, res = nn.scan(
            lm_head_remat,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=num_q,
        )(self.lm_head, None, inputs)
        res = rearrange(res, 'n b c d -> b (n c) d')
        return res

def blockwise_cross_entropy(logits, tokens, valid=None,
                            chunk_size=None, policy=None, prevent_cse=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    valid = valid.astype(jnp.float32)
    logits = jnp.reshape(logits, (-1, logits.shape[-1]))
    tokens = jnp.reshape(tokens, (-1,))
    valid = jnp.reshape(valid, (-1,))

    def _cross_entropy_loss_and_accuracy(logits, tokens, valid):
        valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)

        token_log_prob = jnp.squeeze(
            jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1),
                jnp.expand_dims(tokens, -1),
                axis=-1,
            ),
            -1,
        )
        token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
        correct = jnp.where(
            valid > 0.0,
            jnp.argmax(logits, axis=-1) == tokens,
            jnp.array(False)
        )
        return token_log_prob, correct, valid_text_length
    @partial(jax.checkpoint, prevent_cse=prevent_cse,
             policy=get_gradient_checkpoint_policy(policy))
    def _loss_and_accuracy(carry, args):
        loss, accuracy, num = carry
        logits, tokens, valid = args
        token_log_prob, correct, valid_text_length = \
            _cross_entropy_loss_and_accuracy(logits, tokens, valid)
        loss = loss + jnp.sum(token_log_prob, axis=-1) / valid_text_length
        accuracy = accuracy + jnp.sum(correct, axis=-1) / valid_text_length
        num = num + 1
        return (loss, accuracy, num), None
    num_chunk = logits.shape[0] // chunk_size
    logits = rearrange(logits, '(n c) d -> n c d', c=chunk_size)
    tokens = rearrange(tokens, '(n c) -> n c', c=chunk_size)
    valid = rearrange(valid, '(n c) -> n c', c=chunk_size)
    (loss, accuracy, num), _ = jax.lax.scan(
        _loss_and_accuracy, (0.0, 0.0, 0), xs=(logits, tokens, valid),
        length=num_chunk,
    )
    loss = - loss / num
    accuracy = accuracy / num
    return loss, accuracy

if __name__ == '__main__':
    with jax.profiler.trace('/tmp/prof/blockwise_parallel_simplified'):
        class Model(nn.Module):
            def setup(self):
                self.blocks = [
                    AttentionBlock(
                        q_chunk_size=256,
                        k_chunk_size=256,
                        hidden_size=2048,
                        num_heads=16,
                        rotary_dim=128,
                        intermediate_size=8192,
                        layer_norm_epsilon=1e-5,
                        activation_function="gelu",
                        resid_pdrop=0.0,
                        max_position_embeddings=2048,
                        dtype=jnp.float32,
                        causal=True,
                )
                for _ in range(2)
                ]
            def __call__(self, hidden_states, attention_mask, position_ids):
                for block in self.blocks:
                    hidden_states = block(hidden_states, attention_mask, position_ids)
                return hidden_states

        hidden_states = jnp.zeros((2, 1024, 2048))
        attention_mask = jnp.zeros((2, 1024), dtype=jnp.int32)
        position_ids = jnp.zeros((2, 1024), dtype=jnp.int32)
        model = Model()
        variables = model.init(jax.random.PRNGKey(0), hidden_states, attention_mask, position_ids)
        output = model.apply(variables, hidden_states, attention_mask, position_ids)
        output = output.block_until_ready()
