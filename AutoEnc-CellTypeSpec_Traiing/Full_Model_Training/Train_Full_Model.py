import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn import utils
from enformer_pytorch import Enformer, seq_indices_to_one_hot
from datasets import load_dataset
import math
from pathlib import Path
import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint_sequential
from tqdm import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot
from enformer_pytorch.config_enformer import EnformerConfig
from transformers import PreTrainedModel, PretrainedConfig
from torch.optim.lr_scheduler import LambdaLR

# Constants
SEQUENCE_LENGTH = 114688  # 2**17
TASK_NAME = "cage_prediction"
REFERENCE_GENOME_PATH = "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Downloads/cache/downloads/hg38.fa.gz"
CACHE_DIR = "/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Downloads/cache"

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=SEQUENCE_LENGTH,
    data_files={"reference_genome": REFERENCE_GENOME_PATH},
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# helper methods
SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896
pt_path = '/om2/user/mabdel03/files/Classes/6.8611/Project/Baseline_Model/Batch_Training/Downloads/tf_gammas.pt'
TF_GAMMAS = torch.load(pt_path, weights_only=True)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)
    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def MaybeSyncBatchnorm(is_distributed = None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d

def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()

def pearson_corr_coef(x, y, dim = 1, reduce_dims = (-1,)):
    x_centered = x - x.mean(dim = dim, keepdim = True)
    y_centered = y - y.mean(dim = dim, keepdim = True)
    return F.cosine_similarity(x_centered, y_centered, dim = dim).mean(dim = reduce_dims)

def get_positional_features_exponential(positions, features, seq_len, min_half_life = 3., dtype = torch.float):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device = positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)

def get_positional_features_central_mask(positions, features, seq_len, dtype = torch.float):
    center_widths = 2 ** torch.arange(1, features + 1, device = positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)

def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)

def get_positional_features_gamma(positions, features, seq_len, stddev = None, start_mean = None, eps = 1e-8, dtype = torch.float):
    if not exists(stddev):
        stddev = seq_len / (2 * features)
    if not exists(start_mean):
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device = positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2

    probabilities = gamma_pdf(positions.to(dtype).abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim = -1, keepdim = True)
    return outputs

def get_positional_embed(seq_len, feature_size, device, use_tf_gamma, dtype = torch.float):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)
    assert not use_tf_gamma or seq_len == 1536, 'if using tf gamma, only sequence length of 1536 allowed for now'

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma if not use_tf_gamma else always(TF_GAMMAS.to(device))
    ]

    num_components = len(feature_functions) * 2
    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len, dtype = dtype))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings.to(dtype)

def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim = -1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., :((t2 + 1) // 2)]

# helper classes
from transformers import PretrainedConfig

class EnformerConfig(PretrainedConfig):
    model_type = "enformer"

    def __init__(
        self,
        dim = 1536,
        depth = 11,
        heads = 8,
        output_heads = {
            "human": 5313,
            "mouse": 1643
        },#dict(human = 5313, mouse= 1643),
        target_length = 896,
        attn_dim_key = 64,
        dropout_rate = 0.4,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        use_checkpointing = False,
        use_convnext = False,
        num_downsamples = 7,    # genetic sequence is downsampled 2 ** 7 == 128x in default Enformer - can be changed for higher resolution
        dim_divisible_by = 128,
        use_tf_gamma = False,
        **kwargs,
    ):
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.output_heads = output_heads
        self.target_length = target_length
        self.attn_dim_key = attn_dim_key
        self.dropout_rate = dropout_rate
        self.attn_dropout = attn_dropout
        self.pos_dropout = pos_dropout
        self.use_checkpointing = use_checkpointing
        self.num_downsamples = num_downsamples
        self.dim_divisible_by = dim_divisible_by
        self.use_tf_gamma = use_tf_gamma

        super().__init__(**kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)

        nn.init.dirac_(self.to_attn_logits.weight)
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length
        if target_len == -1:
            return x
        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')
        trim = (target_len - seq_len) // 2
        if trim == 0:
            return x
        return x[:, -trim:trim]

def ConvBlock(dim, dim_out = None, kernel_size = 1, is_distributed = None):
    batchnorm_klass = MaybeSyncBatchnorm(is_distributed = is_distributed)
    return nn.Sequential(
        batchnorm_klass(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_rel_pos_features,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.,
        use_tf_gamma = False
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        self.num_rel_pos_features = num_rel_pos_features
        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_tf_gamma = use_tf_gamma

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = get_positional_embed(n, self.num_rel_pos_features, device, use_tf_gamma = self.use_tf_gamma, dtype = self.to_rel_k.weight.dtype)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        half_dim = config.dim // 2
        filter_list = exponential_linspace_int(half_dim, config.dim,
                                               num=(config.num_downsamples - 1),
                                               divisible_by=config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        encoder_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            encoder_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1)),
                AttentionPool(dim_out, pool_size=2)
            ))

        self.encoder_layers = nn.Sequential(*encoder_layers)

        self.attention = Attention(
            dim=filter_list[-1],
            heads=8,
            dim_key=64,
            dim_value=filter_list[-1] // 8,
            num_rel_pos_features=filter_list[-1] // 8
        )

    def forward(self, x):
        x = self.encoder_layers(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.attention(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        half_dim = config.dim // 2
        filter_list = exponential_linspace_int(half_dim, config.dim,
                                               num=(config.num_downsamples - 1),
                                               divisible_by=config.dim_divisible_by)
        filter_list = [half_dim, *filter_list]

        self.attention = Attention(
            dim=filter_list[-1],
            heads=8,
            dim_key=64,
            dim_value=filter_list[-1] // 8,
            num_rel_pos_features=filter_list[-1] // 8
        )

        decoder_layers = []
        reversed_filters = list(reversed(filter_list))
        for dim_in, dim_out in zip(reversed_filters[:-1], reversed_filters[1:]):
            decoder_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBlock(dim_in, dim_out, kernel_size=5),
                Residual(ConvBlock(dim_out, dim_out, 1))
            ))

        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.attention(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.decoder_layers(x)
        return x
    
# Shallow Enformer
class ShallowEnformer(PreTrainedModel):
    config_class = EnformerConfig
    base_self_prefix = "enformer"

    @staticmethod
    def from_hparams(**kwargs):
        return ShallowEnformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        super().__init__(config)
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            Residual(ConvBlock(half_dim)),
            AttentionPool(half_dim, pool_size=2)
        )

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        use_tf_gamma = config.use_tf_gamma
        self.use_tf_gamma = use_tf_gamma

        transformer = []
        for _ in range(config.depth):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    Attention(
                        config.dim,
                        heads=config.heads,
                        dim_key=config.attn_dim_key,
                        dim_value=config.dim // config.heads,
                        dropout=config.attn_dropout,
                        pos_dropout=config.pos_dropout,
                        num_rel_pos_features=config.dim // config.heads,
                        use_tf_gamma=use_tf_gamma
                    ),
                    nn.Dropout(config.dropout_rate)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(config.dim),
                    nn.Linear(config.dim, config.dim * 2),
                    nn.Dropout(config.dropout_rate),
                    nn.ReLU(),
                    nn.Linear(config.dim * 2, config.dim),
                    nn.Dropout(config.dropout_rate)
                ))
            ))
        self.transformer = nn.Sequential(*transformer)

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        self.final_pointwise = nn.Sequential(
            Rearrange('b n d -> b d n'),
            ConvBlock(self.dim, twice_dim, 1),
            Rearrange('b d n -> b n d'),
            nn.Dropout(config.dropout_rate / 8),
            GELU()
        )

        self._trunk = nn.Sequential(
            self.stem,
            self.encoder,
            #Rearrange('b d n -> b n d'),
            self.transformer,
            self.crop_final,
            self.final_pointwise
        )

        self.add_heads(**config.output_heads)
        self.use_checkpointing = config.use_checkpointing

    def add_heads(self, **kwargs):
        self.output_heads = kwargs
        self._heads = nn.ModuleDict(map_values(lambda features: nn.Sequential(
            nn.Linear(self.dim * 2, features),
            nn.Softplus()
        ), kwargs))

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    def trunk_checkpointed(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        #x = rearrange(x, 'b d n -> b n d')
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        x = self.final_pointwise(x)
        return x

    def forward(
        self,
        x,
        target=None,
        return_corr_coef=False,
        return_embeddings=False,
        return_only_embeddings=False,
        head=None,
        target_length=None,
    ):
        if isinstance(x, str):
            x = [x]  # Wrap string in a list
        if isinstance(x, list):
            x = str_to_one_hot(x)  # Convert list of sequences to one-hot encoding
        elif isinstance(x, torch.Tensor) and x.dtype == torch.long:
            x = seq_indices_to_one_hot(x)  # Convert indices to one-hot encoding

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be a tensor or convertible to a tensor. Got {type(x)}.")

        x = x.to(self.device)  # Ensure tensor is on the correct device

        no_batch = x.ndim == 2
        if no_batch:
            x = rearrange(x, "... -> () ...")

        if exists(target_length):
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        x = trunk_fn(x)

        if no_batch:
            x = rearrange(x, "() ... -> ...")

        if return_only_embeddings:
            return x

        out = map_values(lambda fn: fn(x), self._heads)

        if exists(head):
            assert head in self._heads, f"head {head} not found"
            out = out[head]

        if exists(target):
            assert exists(head), "head must be passed in if one were to calculate loss directly with targets"
            if return_corr_coef:
                return pearson_corr_coef(out, target)
            return poisson_loss(out, target)

        if return_embeddings:
            return out, x
        return out


def from_pretrained(name, use_tf_gamma = None, **kwargs):
    enformer = Enformer.from_pretrained(name, **kwargs)
    if name == 'EleutherAI/enformer-official-rough':
        use_tf_gamma = default(use_tf_gamma, True)
        for module in enformer.modules():
            if isinstance(module, Attention):
                module.use_tf_gamma = use_tf_gamma
    return enformer


# loading autoencoder into enformer
def load_shallow_enformer_with_autoencoder(config_kwargs, autoencoder_weights_path):
    # Load the configuration for the ShallowEnformer
    config = EnformerConfig(**config_kwargs)
    shallow_enformer = ShallowEnformer(config)

    # Load the pre-trained weights
    pretrained_autoencoder_weights = torch.load(autoencoder_weights_path, map_location='cpu')

    # Filter and load encoder weights
    encoder_weights = {
        key[len("encoder."):]: value
        for key, value in pretrained_autoencoder_weights.items()
        if key.startswith("encoder.")
    }
    shallow_enformer.encoder.load_state_dict(encoder_weights, strict=False)

    # Filter and load decoder weights
    decoder_weights = {
        key[len("decoder."):]: value
        for key, value in pretrained_autoencoder_weights.items()
        if key.startswith("decoder.")
    }
    shallow_enformer.decoder.load_state_dict(decoder_weights, strict=False)

    return shallow_enformer


config_kwargs = {
    # Add configuration details here
    # Example: 'dim': 512, 'depth': 6, etc.
    'dim':1536,
    'depth':11,
    'heads':8,
    'output_heads':dict(human=50),
    'target_length':896,

}

autoencoder_weights_path = '/om2/user/mabdel03/files/Classes/6.8611/Project/New_Model/Autoencoder_Training/Models/Trained_AutoEnc_Params.pth'

shallow_enformer = load_shallow_enformer_with_autoencoder(config_kwargs, autoencoder_weights_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
shallow_enformer = shallow_enformer.to(device)

# training model
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

class DNADataset(Dataset):
    def __init__(self, dataset, sequence_length, mapping):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get single sample
        sample = self.dataset[idx]

        # Get sequence and pad/truncate to required length
        raw_sequence = sample['sequence']
        if len(raw_sequence) < self.sequence_length:
            # Pad sequence if too short
            raw_sequence = raw_sequence + 'N' * (self.sequence_length - len(raw_sequence))
        elif len(raw_sequence) > self.sequence_length:
            # Truncate if too long
            raw_sequence = raw_sequence[:self.sequence_length]

        # Convert to indices
        sequence = torch.tensor([self.mapping[base] for base in raw_sequence], dtype=torch.long)

        # Get labels if they exist
        labels = sample.get('labels', None)
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.float32)

        return {
            'sequence': raw_sequence,
            'labels': labels
        }

mapping = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4}
required_sequence_length = 114688

batch_size = 4
num_epochs = 10
learning_rate = 0.001

train_loader = DataLoader(
    DNADataset(train_dataset, required_sequence_length, mapping),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    DNADataset(val_dataset, required_sequence_length, mapping),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

def warmup_lr_scheduler(optimizer, warmup_steps, target_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0  # Maintain target_lr after warmup
    return LambdaLR(optimizer, lr_lambda)


model = shallow_enformer

# Adjusted optimizer with lower initial learning rate
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,  # Target learning rate
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler
warmup_steps = 5000
scheduler = warmup_lr_scheduler(optimizer, warmup_steps, target_lr=0.0005)

# Gradient clipping
max_norm = 0.2

# Training and Validation Loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        #one_hot_sequence = one_hot_sequence.cuda()
        #target = target.cuda()

        one_hot_sequence = str_to_one_hot(batch['sequence']).to(device, non_blocking=True).transpose(1, 2)
        target = batch['labels'].to(device, non_blocking=True)
        # Forward pass
        predictions = model(
            one_hot_sequence,
            head='human',
        )

        # Apply log normalization
        predictions_log_norm = torch.log1p(predictions)
        target_log_norm = torch.log1p(target)

        # Compute loss (e.g., MSE or another loss)
        loss = F.mse_loss(predictions_log_norm, target_log_norm)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # Update learning rate
        scheduler.step()

        total_train_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            one_hot_sequence = str_to_one_hot(batch['sequence']).to(device, non_blocking=True).transpose(1, 2)
            target = batch['labels'].to(device, non_blocking=True)
            target = target.cuda()

            # Forward pass
            predictions = model(
                one_hot_sequence,
                head='human',
            )

            # Apply log normalization
            predictions_log_norm = torch.log1p(predictions)
            target_log_norm = torch.log1p(target)

            # Compute validation loss
            val_loss = F.mse_loss(predictions_log_norm, target_log_norm)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f"/om2/user/mabdel03/files/Classes/6.8611/Project/New_Model/Full_Model_Training/Models/checkpoint_epoch_{epoch+1}.pth")

print("Training Complete")

# Save final model
torch.save(model.module.state_dict(), "/om2/user/mabdel03/files/Classes/6.8611/Project/New_Model/Full_Model_Training/Models/Full_Model_Params.pth")
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
}, "/om2/user/mabdel03/files/Classes/6.8611/Project/New_Model/Full_Model_Training/Models/Full_Model_State.pth")
