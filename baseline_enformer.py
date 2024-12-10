import torch
from enformer_pytorch import Enformer, seq_indices_to_one_hot

model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 50, mouse = 50),
    target_length = 896,
).cuda()