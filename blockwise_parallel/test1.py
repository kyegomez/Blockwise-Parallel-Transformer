import torch
import torch.nn as nn

class BlockwiseParallelTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, num_heads, num_query_blocks, num_kv_blocks):
        super(BlockwiseParallelTransformer, self).__init__()
        self.query_blocks = num_query_blocks
        self.kv_blocks = num_kv_blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.query_layer = nn.Linear(input_dim, num_heads * head_dim)
        self.key_layer = nn.Linear(input_dim, num_heads * head_dim)
        self.value_layer = nn.Linear(input_dim, num_heads * head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim),
        )

    def forward(self, x):
        b, n, _ = x.shape
        q_chunk_size = n // self.query_blocks
        kv_chunk_size = n // self.kv_blocks

        outputs = torch.zeros_like(x)
        for q_idx in range(self.query_blocks):
            q_chunk_start = q_idx * q_chunk_size
            q_chunk_end = (q_idx + 1) * q_chunk_size

            q_ = self.query_layer(x[:, q_chunk_start:q_chunk_end])
            q = q_.view(b, q_chunk_size, self.num_heads, self.head_dim)
            q = q / torch.sqrt(torch.tensor(self.head_dim).float())

            attn_numerator = torch.zeros_like(q)
            attn_denominator = torch.zeros_like(q)

            for kv_idx in range(self.kv_blocks):
                kv_chunk_start = kv_idx * kv_chunk_size
                kv_chunk_end = (kv_idx + 1) * kv_chunk_size

                k_ = self.key_layer(x[:, kv_chunk_start:kv_chunk_end])
                v_ = self.value_layer(x[:, kv_chunk_start:kv_chunk_end])

                k = k_.view(b, kv_chunk_size, self.num_heads, self.head_dim)
                v = v_.view(b, kv_chunk_size, self.num_heads, self.head_dim)

                attn_weight = torch.einsum('bhqd,bkhd->bhqk', q, k)

                max_score, _ = torch.max(attn_weight, dim=-1, keepdim=True)
                exp_weight = torch.exp(attn_weight - max_score)
                attn_numerator += torch.einsum('bhqv,bvhf->bhqf', exp_weight, v)
                attn_denominator += exp_weight.sum(dim=-1, keepdim=True)

            attn_out = (attn_numerator / attn_denominator)
            attn_out = attn_out.contiguous().view(-1, self.num_heads * self.head_dim)
            ffn_out = self.ffn(attn_out + x[:, q_chunk_start:q_chunk_end])
            outputs[:, q_chunk_start:q_chunk_end] = ffn_out + attn_out + x[:, q_chunk_start:q_chunk_end]

        return outputs

    
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
model = BlockwiseParallelTransformer(input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size)


#pass the input sequence to the module to get the output
output = model(x)

print(output.shape)