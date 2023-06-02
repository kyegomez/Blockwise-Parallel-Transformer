import torch 
from torch import nn

class BlockwiseParallelTransformerAttention(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, max_seq_len, block_size):
        super(BlockwiseParallelTransformerAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        
        self.query_projection = nn.Linear(input_size, num_heads * hidden_size)
        self.key_projection = nn.Linear(input_size, num_heads * hidden_size)
        self.value_projection = nn.Linear(input_size, num_heads * hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(num_heads * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_heads * hidden_size)
        )
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(num_heads * hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        num_blocks = seq_len // self.block_size
        query_blocks = x[:, :num_blocks*self.block_size, :].view(batch_size, num_blocks, self.block_size, input_size)
        key_value_blocks = x[:, :num_blocks*self.block_size, :].view(batch_size, num_blocks, self.block_size, input_size)
        
        for i in range(self.num_layers):
            for outer in range(num_blocks):
                query = self.query_projection(query_blocks[:, outer, :, :])
                for inner in range(num_blocks):
                    key = self.key_projection(key_value_blocks[:, inner, :, :])
                    value = self.value_projection(key_value_blocks[:, inner, :, :])
                    
                    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
                    attention_weights = nn.functional.softmax(attention_scores, dim=-1)
                    attention_output = torch.matmul(attention_weights, value)
                    
                    if inner == 0:
                        blockwise_attention_output = attention_output
                    else:
                        blockwise_attention_output = torch.cat((blockwise_attention_output, attention_output), dim=2)
                
                blockwise_attention_output = blockwise_attention_output / torch.sqrt(torch.tensor(blockwise_attention_output.size(-1), dtype=torch.float32))
                feedforward_output = self.feedforward(blockwise_attention_output)
                residual_output = query_blocks[:, outer, :, :] + feedforward_output
                query_blocks[:, outer, :, :] = self.layer_norm1(residual_output)
                
            query_blocks = self.layer_norm2(query_blocks.view(batch_size, num_blocks*self.block_size, self.num_heads*self.hidden_size)).view(batch_size, num_blocks, self.block_size, self.num_heads*self.hidden_size)
            
        return query_blocks.view(batch_size, seq_len, self.num_heads*self.hidden_size)
    

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