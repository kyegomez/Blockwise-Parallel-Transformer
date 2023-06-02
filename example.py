import jax 
import jax.numpy as jnp
from jax import random
from blockwise_parallel import BlockwiseParallelTransformerAttention



#hyperparams
input_size = 512
num_heads = 8
hidden_size = 512
num_layers = 6
max_seq_len = 1024
block_size = 64

#create random input sequence
key = random.PRNGKey(0)
x = random.normal(key, (1, max_seq_len, input_size))

#create instance
attention = BlockwiseParallelTransformerAttention(input_size,
                                                  num_heads,
                                                  hidden_size,
                                                  num_layers,
                                                  max_seq_len,
                                                  block_size)

##compute the output of the attention
output = attention(x)

#print the shape of the output
print(output.shape)