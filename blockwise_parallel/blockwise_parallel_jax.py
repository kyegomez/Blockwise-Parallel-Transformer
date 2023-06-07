import jax
import jax.numpy as jnp
from jax import nn, lax
from jax.experimental.stax import Dense

class BlockwiseParallelTransformerAttention:
    def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.dim_per_head = hidden_size // num_heads

        self.query_chunk_size = max_seq_len // block_size
        self.key_value_chunk_size = max_seq_len // block_size
        self.num_query_chunks = (max_seq_len + self.query_chunk_size - 1) // self.query_chunk_size
        self.num_key_value_chunks = (max_seq_len + self.key_value_chunk_size - 1) // self.key_value_chunk_size

        self.query_position_ids = jnp.arange(max_seq_len)
        self.key_value_position_ids = jnp.arange(max_seq_len)

        self.query_blocks = Dense(hidden_size, name='query')
        self.key_blocks = Dense(hidden_size, name='key')
        self.value_blocks = Dense(hidden_size, name='value')
        self.feedforward = Dense(hidden_size, name='feedforward')

    def _chunk_bias_fn(self, query_chunk_idx, key_chunk_idx):
        start = key_chunk_idx * self.key_value_chunk_size
        end = (key_chunk_idx + 1) * self.key_value_chunk_size
        bias_chunk = jnp.zeros((self.num_heads, self.query_chunk_size, self.key_value_chunk_size))
        bias_chunk = lax.dynamic_update_slice(bias_chunk, jnp.ones((self.num_heads, self.query_chunk_size, end - start)), (slice(None), slice(None), slice(start, end)))
        bias_chunk = jnp.expand_dims(bias_chunk, axis=0)
        bias_chunk = jnp.tile(bias_chunk, (query_chunk_idx.shape[0], 1, 1, 1))
        return bias_chunk

    def _query_block(self, input_chunk, query_chunk_idx):
        query_chunk = self.query_blocks(input_chunk)
        query_chunk = query_chunk / jnp.sqrt(query_chunk.shape[-1])
        return query_chunk

    def _key_value_blocks(self, carry, args):
        kv_chunk, key_chunk_idx, kv_position_ids_chunk = args
        query_chunk, query_chunk_idx = carry
        key_chunk = self.key_blocks(kv_chunk)
        value_chunk = self.value_blocks(kv_chunk)
        attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)
        bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
        bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
        attn_weights = attn_weights + bias_chunk
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum('bqhv,bvhf->bqhf', exp_weights, value_chunk)
        numerator = jax.lax.dynamic_update_slice(query_chunk, exp_values, (slice(None), key_chunk_idx, slice(None), slice(None)))
        denominator = jax.lax.dynamic_update_slice(query_chunk, exp_weights.sum(axis=-1, keepdims=True), (slice(None), key_chunk_idx, slice(None), slice(None)))
        return (numerator, denominator), None
    
    def __call__(self, x, deterministic=True):
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"Input size must be {self.input_size} but got {input_size}"

        query_chunks = x.reshape(batch_size, self.num_query_chunks, self.query_chunk_size, input_size)
        query_chunks = self.query_blocks(query_chunks)

        query_chunks = query_chunks / jnp.sqrt(query_chunks.shape[-1])
        query_position_ids = jnp.tile(self.query_position_ids, (batch_size, 1))
        query_position_ids = query_position_ids.reshape(batch_size, self.num_query_chunks, self.query_chunk_size)
        query_position_ids = jax.lax.dynamic_slide(query_position_ids, (0, 0, 0), (batch_size, self.num_query_chunks, self.query_chunk_size - 1))
        query_position_ids = jnp.concatenate([query_position_ids, jnp.ones((batch_size, self.num_query_chunks, 1)) * (self.max_seq_len - 1)], axis=-1)
        query_position_ids = query_position_ids.astype(jnp.int32)

        key_value_chunks = x.reshape(batch_size, self.num_key_value_chinks, self.key_value_chunk_size, input_size)
        key_value_chunks = jax.lax.stop_gradient(key_value_chunks) if deterministic else key_value_chunks
        key_value_position_ids = jnp.tile(self.key_value_position_ids, (batch_size, 1))
        key_value_position_ids = key_value_position_ids.reshape(batch_size, self.num_value_chunks, self.key_value_chunk_size)
        key_value_position_ids = jax.lax.dynamic_slice(key_value_position_ids, (0, 0, 0), (batch_size, self.num_key_value_chunks, self.key_value_chunk_size - 1))
        key_value_position_ids = jnp.concatenate([key_value_position_ids, jnp.ones((batch_size, self.num_key_value_chunks, 1)) * (self.max_seq_len - 1)], axis=-1)
        key_value_position_ids = key_value_position_ids.astype(jnp.int32)

        query_blocks = jax.lax.map(self._query_block, query_chunks, jnp.arange(self.num_query_chunks))
        query_blocks = query_blocks.reshape(batch_size, self.num_query_chunks, self.num_heads, self.query_chunk_size, self.dim_per_head)
        query_blocks = jnp.moveaxis(query_blocks, 2, 3)


        key_value_blocks = key_value_chunks.reshape(batch_size, self.num_key_value_chunks, self.num_heads, self.key_value_chunk_size, self.dim_per_head)
        key_value_blocks = jnp.moveaxis(key_value_blocks, 2, 3)

        carry = (query_blocks, None)
        key_value_blocks = jax.lax.scan(self._key_value_blocks, carry, (key_value_blocks, jnp.arange(self.num_key_value_chunks), key_value_position_ids))[0][0]

        key_value_blocks = jnp.moveaxis(key_value_blocks, 2, 3)
        key_value_blocks = key_value_blocks.reshape(batch_size, self.num_key_value_chunks, self.key_value_chunk_size, self.hidden_size)

        output = jax.lax.map(lambda x: self.feedforward(x.reshape(-1, self.hidden_size)), key_value_blocks)
        output = output.reshape(batch_size, seq_len, self.hidden_size)

        return output

    
    
    
    
    




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