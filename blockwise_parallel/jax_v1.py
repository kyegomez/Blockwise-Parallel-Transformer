import jax
import jax.numpy as jnp
from jax import nn, lax

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

        self.query_blocks = nn.Dense(hidden_size, name='query')
        self.key_blocks = nn.Dense(hidden_size, name='key')
        self.value_blocks = nn.Dense(hidden_size, name='value')
        self.feedforward = nn.Dense(hidden_size, name='feedforward')

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
        assert input_size == self.input_size, f'Input size must be {self.input_size}, but got {input_size}'

        query_chunks = x.reshape(batch_size, self.num_query_chunks, self.query_chunk_size, input_size)
        query_chunks = self.query_blocks(query_chunks)
        query_chunks = query_chunks / jnp.sqrt(query_chunks.shape[-1])

        kv_chunks = x.reshape(batch_size, self.num_key_value_chunks, self.key_value_chunk_size, input_size)
        kv_chunks = self.key_blocks(kv_chunks), self.value_blocks(kv_chunks)

        init_carry = (jnp.zeros((batch_size, self.query_chunk_size, self.num_heads, self.dim_per_head)),
                      jnp.zeros((batch_size, self.query_chunk_size, self.num_heads, self.dim_per_head)),
                      (-jnp.inf) * jnp.ones((batch_size, self.query_chunk_size, self.num_heads, 1)))
        
        for key_chunk_idx in range(self.num_key_value_chunks):
            for key_chunk_idx in range(self.num_key_value_chunks):
                key_value_chunk = kv_chunks[:, key_chunk_idx]
                key_value_position_ids_chunk = self.key_value_position_ids[key_chunk_idx * self.key_value_chunk_size:(key_chunk_idx + 1) * self.key_value_chunk_size]
                carry, _ = lax.scan(self._key_value_blocks, carry, (key_value_chunk, key_chunk_idx, key_value_position_ids_chunk))

            numerator, denominator, bias = carry
            attn_weights = numerator / denominator
            attn_weights = jax.lax.dynamic_update_slice(attn_weights, bias, (slice(None), slice(None), slice(None), 0))
            attn_weights = nn.softmax(attn_weights, axis=-2)
            attn_weights = jax.lax.dynamic_update_slice(attn_weights, jnp.zeros_like(bias), (slice(None), slice(None), slice(None), 0))

            value_chunk = jnp.einsum('bqhv,bvhf->bqhf', attn_weights, kv_chunks)
            value_chunk = value_chunk.reshape(batch_size, self.num_heads * self.query_chunk_size, self.dim_per_head)
            value_chunk = self.feedforward(value_chunk)
            value_chunk = value_chunk.reshape(batch_size, self.num_heads, self.query_chunk_size, self.dim_per_head)
            value_chunk = jnp.moveaxis(value_chunk, 1, 2)
            if query_chunk_idx == 0:
                output = value_chunk
            else:
                output = jnp.concatenate([output, value_chunk], axis=2)

        output = output.reshape(batch_size, seq_len, self.hidden_size)
        return output

        # def _key_value_blocks(cell, carry, args):
        #     kv_chunk, key_chunk_idx, kv_position_ids_chunk = args
        #     query_chunk, query_chunk_idx = carry
        #     key_chunk = self.key_blocks(kv_chunk)
        #     value_chunk = self.value_blocks(kv_chunk)
        #     attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk)
        #     bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
        #     bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
        #     attn_weights = attn_weights + bias_chunk
        #     max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        #     exp_weights = jnp.exp(attn_weights - max_score)
        #     exp_values = jnp.einsum('bqhv,bvhf->bqhf', exp_weights, value_chunk)
        #     numerator = jax.lax.dynamic_update_slice(query_chunk, exp_values, (slice(None), key_chunk_idx, slice(None), slice(None)))
        #     denominator = jax.lax.dynamic_update_slice(query_chunk, exp_weights.sum(axis=-1, keepdims=True), (slice(None), key_chunk_idx, slice(None), slice(None)))
        #     return (numerator, denominator), None
        
        # for query_chunk_idx in range(self.num_query_chunks):
        #     query_chunk = self._query_block(query_chunks[:, query_chunk_idx], query_chunk_idx)
        #     for key_value_chunk_idx in range(self.num_key_value_chunks):
        #         kv_chunk = kv_chunks[:, key_value_chunk_idx, :, :]
        #         init_carry = (query_chunk, query_chunk_idx)
        #         (numerator, denominator), _ = lax.scan(_key_value_blocks, init_carry, (kv_chunk, key_value_chunk_idx))
        #     attention_output_chunk = numerator / denominator 
        #     attention_output_chunk = self.feedforward(attention_output_chunk)
        #     query_chunk = query_chunks[:, query_chunk_idx]
        #     attention_output_chunk = attention_output_chunk + query_chunk
        #     attention_output_chunk = nn.LayerNorm(attention_output_chunk)
        #     query_chunks = jax.lax.dynamic_update_slice(query_chunks, attention_output_chunk, (slice(None), query_chunk_idx, slice(None), slice(None)))
        
        # attention_output = query_chunks.reshape(batch_size, seq_len, self.hidden_size)
        # return attention_output
        

        

