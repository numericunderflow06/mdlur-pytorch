import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# helper modules (guess) 

# 1. CategoricalDenseModel
class CategoricalDenseModel(nn.Module):
    """
    Builds embeddings for categorical features given a vocab_size_dict
    and passes the concatenated embeddings through an MLP.
    Expected input: a dict mapping feature names to LongTensor (batch,).
    """
    def __init__(self, vocab_size_dict, embed_dim=8, hidden_dims=[128, 64]):
        super().__init__()
        self.vocab_size_dict = vocab_size_dict
        self.embed_layers = nn.ModuleDict({
            key: nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for key, vocab_size in vocab_size_dict.items()
        })
        input_dim = embed_dim * len(vocab_size_dict)
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LeakyReLU(inplace=True))
            input_dim = h
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x is a dict with keys matching self.vocab_size_dict
        embedded = [self.embed_layers[k](x[k]) for k in self.vocab_size_dict.keys()]
        # Each is (batch, embed_dim); concatenate along last dim.
        x_cat = torch.cat(embedded, dim=-1)
        return self.mlp(x_cat)

# 2. AutoEncoder 
class AutoEncoder(nn.Module):
    """
    A simple auto-encoder that maps an input vector (of known dimension)
    to a lower-dimensional representation.
    """
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.encoder(x)

# 3. SkipAutoEncoder
class SkipAutoEncoder(nn.Module):
    """
    A feed-forward network with residual connections.
    Uses a LazyLinear for the first layer so that the input dimension is inferred.
    """
    def __init__(self, init_channel_dim, depth, output_dim):
        super().__init__()
        self.initial = nn.LazyLinear(init_channel_dim)
        self.depth = depth
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(init_channel_dim, init_channel_dim),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(depth)
        ])
        self.final = nn.Linear(init_channel_dim, output_dim)
    def forward(self, x):
        x = self.initial(x)
        for block in self.blocks:
            x = x + block(x)
        return self.final(x)

# 4. WeightedSum
class WeightedSum(nn.Module):
    """
    Learns a set of weights to compute a weighted sum of input tensors.
    Default assumes two inputs.
    """
    def __init__(self, num_inputs=2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
    def forward(self, inputs):
        # inputs: list of tensors (all same shape)
        weights = F.softmax(self.weights, dim=0)
        out = sum(w * inp for w, inp in zip(weights, inputs))
        return out

# Modules for Time-Series Modeling        

# pos encoding
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

# transformer block
class TransformerBlock(nn.Module):
    """
    A single transformer encoder block: multi-head self-attention,
    residual connections, and a feed-forward network.
    """
    def __init__(self, d_model, n_heads, ff_hidden_dim, dropout=0.1, use_flash_attention=False):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout, use_flash_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# attention
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention. Optionally uses torch.nn.functional.scaled_dot_product_attention
    if use_flash_attention is True.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash_attention=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Create mask of shape (batch, seq_len, seq_len) if provided
            if mask is not None:
                attn_mask = ~mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            else:
                attn_mask = None
            out = F.scaled_dot_product_attention(q, k, v,
                                                   attn_mask=attn_mask,
                                                   dropout_p=self.dropout.p,
                                                   is_causal=False)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        return self.fc(out)

# feedforward
class FeedForward(nn.Module):
    """
    Point-wise feed-forward network.
    """
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
      
#encoder
class TransformerEncoder(nn.Module):
    """
    Processes an input sequence (batch, window_length, feature_dim) by optionally adding
    time2vec features, applying several Transformer blocks, mean-pooling, and then an MLP.
    """
    def __init__(self, window_length, feature_dim, dense_layers, trans_output_dim,
                 add_time2vec=True, additional_dropout=False, attention_layer_num=3):
        super().__init__()
        self.add_time2vec = add_time2vec
        if add_time2vec:
            self.time2vec = nn.Linear(1, feature_dim)
        # Create a stack of transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=feature_dim, n_heads=2, ff_hidden_dim=feature_dim*2, dropout=0.1)
            for _ in range(attention_layer_num)
        ])
        layers = []
        input_dim = feature_dim
        for units in dense_layers:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU(inplace=True))
            if additional_dropout:
                layers.append(nn.Dropout(0.1))
            input_dim = units
        layers.append(nn.Linear(input_dim, trans_output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        # x: (batch, window_length, feature_dim)
        if self.add_time2vec:
            batch_size, window_length, _ = x.size()
            t = torch.linspace(0, 1, steps=window_length, device=x.device).unsqueeze(0).unsqueeze(-1).expand(batch_size, window_length, 1)
            time_features = self.time2vec(t)
            x = x + time_features
        mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = x.mean(dim=1)  # Mean pooling over time
        return self.mlp(x)

class Conv2dUnet(nn.Module):
    """
    A simplified U-Net style module.
    Input: (batch, window_length, feature_dim)
    Internally reshaped to (batch, 1, window_length, feature_dim).
    Outputs a vector of dimension output_dim.
    """
    def __init__(self, window_length, feature_dim, init_channel_dim=16, depth=2, output_dim=1):
        super().__init__()
        self.depth = depth
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        current_channels = 1
        # Down-sampling path
        for i in range(depth):
            out_channels = init_channel_dim * (2 ** i)
            conv = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.down_convs.append(conv)
            current_channels = out_channels
        # Up-sampling path
        for i in range(depth-1, -1, -1):
            out_channels = init_channel_dim * (2 ** i)
            conv = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.up_convs.append(conv)
            current_channels = out_channels
        self.final_conv = nn.Conv2d(current_channels, output_dim, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        # x: (batch, window_length, feature_dim)
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, window_length, feature_dim)
        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        for conv in self.up_convs:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if downs:
                skip = downs.pop()
                if x.size() == skip.size():
                    x = torch.cat([x, skip], dim=1)
            x = conv(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        return x.view(batch_size, -1)

class PortraitStaticModel(nn.Module):
    """
    Extracts the last time step of the portrait input and passes it through an AutoEncoder.
    """
    def __init__(self, window_length, portrait_length, output_dim=16):
        super().__init__()
        self.autoencoder = AutoEncoder(output_dim, input_dim=portrait_length)
    def forward(self, x):
        # x: (batch, window_length, portrait_length)
        x_last = x[:, -1, :]  # (batch, portrait_length)
        return self.autoencoder(x_last)

# Main CUTAWAS Model (PyTorch Adaptation) 

class CUTAWAS(nn.Module):

    def __init__(self, behavior_length, portrait_length, window_length, target_day, vocab_size_dict, **kwargs):
        super().__init__()
        self.behavior_length = behavior_length
        self.portrait_length = portrait_length
        self.window_length = window_length
        self.target_day = target_day
        self.vocab_size_dict = vocab_size_dict
        self.dense_layers = [128, 64]
        self.attention_num = 3
        self.trans_output_dim = 64
        
        # Build user model
        self.user_model = CategoricalDenseModel(vocab_size_dict)
        
        # Portrait static model (uses last time step)
        self.portrait_static_model = PortraitStaticModel(window_length, portrait_length, output_dim=16)
        
        # Portrait time-series models (Conv2dUnet + TransformerEncoder ensemble)
        self.portrait_conv_unet = Conv2dUnet(window_length, portrait_length, init_channel_dim=16, depth=2, output_dim=1)
        self.portrait_transformer_model = TransformerEncoder(window_length, portrait_length, self.dense_layers, self.trans_output_dim,
                                                               add_time2vec=True, additional_dropout=False, attention_layer_num=self.attention_num)
        self.portrait_weighted_sum = WeightedSum(num_inputs=2)
        
        # Behavior time-series models (Conv2dUnet + TransformerEncoder ensemble)
        self.behavior_conv_unet = Conv2dUnet(window_length, behavior_length, init_channel_dim=16, depth=2, output_dim=1)
        self.behavior_transformer_model = TransformerEncoder(window_length, behavior_length, self.dense_layers, self.trans_output_dim,
                                                               add_time2vec=True, additional_dropout=False, attention_layer_num=self.attention_num)
        self.behavior_weighted_sum = WeightedSum(num_inputs=2)
        
        # Final SkipAutoEncoder: ensembles the concatenated outputs into a target-day prediction vector.
        self.skip_autoencoder = SkipAutoEncoder(init_channel_dim=32, depth=2, output_dim=target_day)
        
    def forward(self, inputs):
        # Expecting inputs: [user_input, portrait_input, behavior_input]
        user_input, portrait_input, behavior_input = inputs
        user_out = self.user_model(user_input)
        portrait_static_out = self.portrait_static_model(portrait_input)
        portrait_conv_out = self.portrait_conv_unet(portrait_input)
        portrait_trans_out = self.portrait_transformer_model(portrait_input)
        portrait_ts_out = self.portrait_weighted_sum([portrait_conv_out, portrait_trans_out])
        behavior_conv_out = self.behavior_conv_unet(behavior_input)
        behavior_trans_out = self.behavior_transformer_model(behavior_input)
        behavior_ts_out = self.behavior_weighted_sum([behavior_conv_out, behavior_trans_out])
        # Concatenate all outputs along the last dimension.
        concatenated = torch.cat([user_out, portrait_static_out, portrait_ts_out, behavior_ts_out], dim=-1)
        z = self.skip_autoencoder(concatenated)
        return z

    def get_config(self):
        # Returns configuration as a dict.
        return {
            'behavior_length': self.behavior_length,
            'portrait_length': self.portrait_length,
            'window_length': self.window_length,
            'target_day': self.target_day,
            'vocab_size_dict': self.vocab_size_dict,
            'dense_layers': self.dense_layers,
            'attention_num': self.attention_num,
            'trans_output_dim': self.trans_output_dim
        }

if __name__ == "__main__":
    # Dummy configuration
    behavior_length = 10
    portrait_length = 20
    window_length = 5
    target_day = 7
    vocab_size_dict = {'gender': 3, 'city': 100, 'age': 100}

    model = CUTAWAS(behavior_length, portrait_length, window_length, target_day, vocab_size_dict)
    
    batch_size = 8
    # For user_input, we expect a dict with keys 'gender', 'city', 'age' (each tensor of shape (batch,))
    user_input = {
        'gender': torch.randint(0, 3, (batch_size,)),
        'city': torch.randint(0, 100, (batch_size,)),
        'age': torch.randint(0, 100, (batch_size,))
    }
    # Portrait input: shape (batch, window_length, portrait_length)
    portrait_input = torch.randn(batch_size, window_length, portrait_length)
    # Behavior input: shape (batch, window_length, behavior_length)
    behavior_input = torch.randn(batch_size, window_length, behavior_length)
    
    # Forward pass
    output = model([user_input, portrait_input, behavior_input])
    print("Output shape:", output.shape)  # Expected: (batch, target_day)
