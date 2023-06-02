import torch 
import torch.nn as nn

class BlockwiseParallelTransformerAttention(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
        super(BlockwiseParallelTransformerAttention, self).__init__()
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

        self.query_position_ids = torch.arange(max_seq_len)
        self.key_value_position_ids = torch.arange(max_seq_len)

        self.query_blocks = nn.Linear(input_size, hidden_size, bias=False)
        self.key_blocks = nn.Linear(input_size, hidden_size, bias=False)
        self.value_blocks = nn.Linear(input_size, hidden_size, bias=False)
        self.feedforward = nn.Linear(hidden_size, hidden_size)

    def _chunk_bias_fn(self, query_chunk_idx, key_chunk_idx):
        start = key_chunk_idx * self.key_value_chunk_size
        end = (key_chunk_idx + 1) * self.key_value_chunk_size
        bias_chunk = torch.zeros((self.num_heads, self.query_chunk_size, self.key_value_chunk_size))
        bias_chunk[:, :, start:end] = 1
        bias_chunk = bias_chunk.unsqueeze(0)
        bias_chunk = bias_chunk.repeat(query_chunk_idx.shape[0], 1, 1, 1)
        return bias_chunk
    
    def _query_block(self, input_chunk, query_chunk_idx):
        query_chunk = self.query_blocks(input_chunk)
        query_chunk = query_chunk / torch.sqrt(query_chunk.shape[-1])
        return query_chunk
    
    def _key_value_blocks(self, carry, args):
        kv_chunk, key_chunk_idx, kv_position_ids_chunk = args
        query_chunk, query_chunk_idx = carry
        key_chunk = self.key_blocks(kv_chunk)
        value_chunk = self.value_blocks(kv_chunk)
        attn_weights = torch.einsum('bqhd, bkhd->bqhk', query_chunk. key_chunk)
        bias_chunk = self._chunk_bias_fn(query_chunk_idx, key_chunk_idx)
        bias_chunk = bias_chunk.permute(0, 1, 3, 2)
        attn_weights = attn_weights + bias_chunk
        max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.einsum('bqhv, bvhf->bqhf', exp_weights, value_chunk)
        numerator = query_chunk.clone()
        numerator[:, key_chunk_idx, :, :] = exp_values
        denominator = query_chunk.clone()
        denominator[:, key_chunk_idx, :, :] = exp_weights.sum(dim=-1, keepdim=True)
        return (numerator, denominator), None
    
    def forward(self, x, deterministic=None):
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size, f"Input size must be {self.input_size} but got {input_size}"

        query_chunks = x.reshape(batch_size, self.num_query_chunks, self.query_chunk_size, input_size)
        query_chunks = self.query_blocks(query_chunks)

        query_chunks = query_chunks / torch.sqrt(query_chunks.shape[-1])
        query_position_ids = self.query_position_ids.repeat(batch_size, 1)
        query_position_ids = query_position_ids.reshape(batch_size, self.num_query_chunks, self.query_chunk_size)
        query_position_ids = query_position_ids.roll(shift=-1, dims=-1)
        query_position_ids[:, :, -1] = self.max_seq_len - 1

        key_value_chunks = x.reshape(batch_size, self.num_key_value_chunks, self.key_value_chunk_size, input_size)
        key_value_chunks = key_value_chunks.detach() if deterministic else key_value_chunks
        key_value_position_ids = self.key_value_chunk_position_ids.repeat(batch_size, 1)
        key_value_position_ids = key_value_position_ids[:, :-1, :]
        key_value_position_ids = torch.cat([key_value_position_ids, torch.ones((batch_size, 1, self.key_value_chunk_size)) * (self.max_seq_len -1)], dim=1)

        carry = (query_chunks, None)
        for key_chunk_idx in range(self.num_key_value_chunks):
            kv_chunk = key_value_chunks[:, key_chunk_idx, :, :]
            kv_position_ids_chunk = key_value_position_ids[:, key_chunk_idx, :]
            carry, _ = self._key_value_blocks(carry, (kv_chunk, key_chunk_idx, kv_position_ids_chunk))

        attn_output = carry[0]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.feedforward(attn_output)

        return attn_output
 

    




#inpout sequence
batch_size = 2
seq_len = 1024
input_size = 512
x = torch.randn(batch_size, seq_len, input_size)


#define params
num_heads = 8
hidden_size = 512
num_layers = 6
max_seq_len = 1024
block_size = 64

#crete an instance of blockwise paralel
model = BlockwiseParallelTransformerAttention(input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size)


#pass the input sequence to the module to get the output
output = model(x)

print(output.shape)





















# import torch 
# from torch import nn

# class BlockwiseParallelTransformerAttention(nn.Module):
#     def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
#         super(BlockwiseParallelTransformerAttention, self).__init__()
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.max_seq_len = max_seq_len
#         self.block_size = block_size
        
#         self.query_projection = nn.Linear(input_size, num_heads * hidden_size)
#         self.key_projection = nn.Linear(input_size, num_heads * hidden_size)
#         self.value_projection = nn.Linear(input_size, num_heads * hidden_size)
#         self.feedforward = nn.Sequential(
#             nn.Linear(num_heads * hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_heads * hidden_size)
#         )
#         self.layer_norm1 = nn.LayerNorm(input_size)
#         self.layer_norm2 = nn.LayerNorm(num_heads * hidden_size)
        
#     def forward(self, x):
#         batch_size, seq_len, input_size = x.size()
#         num_blocks = seq_len // self.block_size
#         query_blocks = x[:, :num_blocks*self.block_size, :].view(batch_size, num_blocks, self.block_size, input_size)
#         key_value_blocks = x[:, :num_blocks*self.block_size, :].view(batch_size, num_blocks, self.block_size, input_size)
        
#         for i in range(self.num_layers):
#             for outer in range(num_blocks):
#                 query = self.query_projection(query_blocks[:, outer, :, :])
#                 for inner in range(num_blocks):
#                     key = self.key_projection(key_value_blocks[:, inner, :, :])
#                     value = self.value_projection(key_value_blocks[:, inner, :, :])
                    
#                     attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
#                     attention_weights = nn.functional.softmax(attention_scores, dim=-1)
#                     attention_output = torch.matmul(attention_weights, value)
                    
#                     if inner == 0:
#                         blockwise_attention_output = attention_output
#                     else:
#                         blockwise_attention_output = torch.cat((blockwise_attention_output, attention_output), dim=2)
                
#                 blockwise_attention_output = blockwise_attention_output / torch.sqrt(torch.tensor(blockwise_attention_output.size(-1), dtype=torch.float32))
#                 feedforward_output = self.feedforward(blockwise_attention_output)
#                 residual_output = query_blocks[:, outer, :, :] + feedforward_output
#                 query_blocks[:, outer, :, :] = self.layer_norm1(residual_output)
                
#             query_blocks = self.layer_norm2(query_blocks.view(batch_size, num_blocks*self.block_size, self.num_heads*self.hidden_size)).view(batch_size, num_blocks, self.block_size, self.num_heads*self.hidden_size)
            
#         return query_blocks.view(batch_size, seq_len, self.num_heads*self.hidden_size)
    