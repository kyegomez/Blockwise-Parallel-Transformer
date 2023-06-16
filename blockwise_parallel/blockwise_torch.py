import torch 
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "swish": F.silu,
    "gelu_new": F.gelu,
}

def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / torch.pow(10000, 2 * torch.arange(dim // 2).float() / dim)
    pos_emb = torch.zeros(num_pos, dim)
    pos_emb[:, 0::2] = torch.sin(inv_freq * torch.arange(num_pos).float().unsqueeze(1))
    pos_emb[:, 1::2] = torch.cos(inv_freq * torch.arange(num_pos).float().unsqueeze(1))
    return pos_emb

def rotate_every_two(tensor):
    tensor = tensor.permute(0, 1, 3, 2)
    tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1, 2)
    tensor = tensor.permute(0, 1, 3, 2)
    tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1)
    return tensor

def apply_rotary_pos_emb(tensor, sicncos):
    sin_emb, cos_emb = sicncos
    sin_emb = sin_emb.unsqueeze(2).repeat(1, 1, tensor.size(2) // 2)
    cos_emb = cos_emb.unsqueeze(2).repeat(1, 1, tensor.size(2) // 2)
    tensor_rotated = tensor.reshape(tensor.size(0), tensor.size(1), -1, 2)
    tensor_rotated = tensor_rotated.permute(0, 1, 3, 2)
    tensor_rotated[:, :, 0, :] = tensor.rotated[:, :, 0, :] * cos_emb - tensor_rotated[:, :, 1, :] * sin_emb
    tensor_rotated[:, :, 1, :] = tensor_rotated[:, :, 0, :] * sin_emb + tensor_rotated[:, :, 1, :] * cos_emb
    tensor = tensor_rotated.permute(0, 1, 3, 2)
    tensor = tensor.reshape(tensor.size(0), tensor.size(1), -1)
    return tensor
