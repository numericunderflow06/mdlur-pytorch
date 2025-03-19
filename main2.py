import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

#############################################
# 1. DATA PROCESSING
#############################################

def load_and_preprocess_data(
    portrait_csv="sample_data_player_portrait.csv",
    behavior_csv="sample_data_behavior_sequence.csv",
    social_csv="sample_data_social_network.csv",
    label_csv="sample_data_label.csv",
    window_length=5
):
    """
    Loads the 4 CSV files and processes them to output:
      - user_input_dict: dict of user-level categorical features (uid is not used as a feature)
      - portrait_ts_tensor: tensor [N, window_length, portrait_dim]
      - behavior_ts_tensor: tensor [N, window_length, behavior_dim]
      - social_tensor: tensor [N, 2] with simple social features
      - churn_label_tensor: tensor [N] (target for churn prediction)
      - payment_label_tensor: tensor [N] (target for payment prediction)
    """
    # Read CSV files
    df_portrait = pd.read_csv(portrait_csv)
    df_behavior = pd.read_csv(behavior_csv)
    df_social = pd.read_csv(social_csv)
    df_label = pd.read_csv(label_csv)

    # --- Social features ---
    df_social['edge_count'] = 1
    social_outgoing = df_social.groupby('src_uid')['edge_count'].sum().reset_index()
    social_outgoing.columns = ['uid', 'social_out_degree']
    df_label = df_label.merge(social_outgoing, on='uid', how='left')
    df_label['social_out_degree'] = df_label['social_out_degree'].fillna(0)
    social_incoming = df_social.groupby('dst_uid')['edge_count'].sum().reset_index()
    social_incoming.columns = ['uid', 'social_in_degree']
    df_label = df_label.merge(social_incoming, on='uid', how='left')
    df_label['social_in_degree'] = df_label['social_in_degree'].fillna(0)
    df_label['social_out_degree'] = df_label['social_out_degree'].astype(float)
    df_label['social_in_degree'] = df_label['social_in_degree'].astype(float)

    # --- Portrait time-series ---
    df_portrait['ds'] = pd.to_datetime(df_portrait['ds'])
    df_portrait = df_portrait.sort_values(['uid', 'ds'])
    def get_last_n_rows(group, n=window_length):
        return group.tail(n)
    df_portrait_lastN = df_portrait.groupby('uid', group_keys=False).apply(get_last_n_rows).reset_index(drop=True)
    # Look for numeric columns; adjust prefix if necessary (e.g., "feature_" instead of "feature")
    numeric_cols = [c for c in df_portrait.columns if c.startswith('feature')]
    portrait_dict = defaultdict(list)
    for uid, grp in df_portrait_lastN.groupby('uid'):
        grp = grp.sort_values('ds')
        feats_2d = grp[numeric_cols].values.tolist()
        if len(feats_2d) < window_length:
            pad_count = window_length - len(feats_2d)
            pad_array = [[0.0]*len(numeric_cols) for _ in range(pad_count)]
            feats_2d = pad_array + feats_2d
        portrait_dict[uid] = feats_2d

    # --- Behavior time-series ---
    df_behavior['ds'] = pd.to_datetime(df_behavior['ds'])
    df_behavior = df_behavior.sort_values(['uid', 'ds'])
    def seq_to_features(seq_str):
        arr = [int(x) for x in seq_str.split(',')] if pd.notnull(seq_str) else []
        if len(arr) == 0:
            return [0.0, 0.0, 0.0]
        return [float(sum(arr)), float(max(arr)), float(len(arr))]
    df_behavior['behavior_feat'] = df_behavior['seq'].apply(seq_to_features)
    df_behavior[['beh_sum','beh_max','beh_len']] = pd.DataFrame(df_behavior['behavior_feat'].tolist(), index=df_behavior.index)
    def get_last_n_rows_behavior(group, n=window_length):
        return group.tail(n)
    df_behavior_lastN = df_behavior.groupby('uid', group_keys=False).apply(get_last_n_rows_behavior).reset_index(drop=True)
    behavior_dict = defaultdict(list)
    behavior_cols = ['beh_sum','beh_max','beh_len']
    for uid, grp in df_behavior_lastN.groupby('uid'):
        grp = grp.sort_values('ds')
        feats_2d = grp[behavior_cols].values.tolist()
        if len(feats_2d) < window_length:
            pad_count = window_length - len(feats_2d)
            pad_array = [[0.0]*len(behavior_cols) for _ in range(pad_count)]
            feats_2d = pad_array + feats_2d
        behavior_dict[uid] = feats_2d

    # --- Merge and build final arrays ---
    final_uids = df_label['uid'].unique().tolist()
    # Create user-level categorical features (uid itself is not used)
    user_static_dict = {}
    for uid in final_uids:
        user_static_dict[uid] = {
            'region': uid % 5,
            'segment': uid % 10
        }

    portrait_list = []
    behavior_list = []
    social_list = []
    churn_list = []
    payment_list = []
    user_static_list = []

    for uid in final_uids:
        if uid not in portrait_dict:
            portrait_feats = [[0.0]*len(numeric_cols) for _ in range(window_length)]
        else:
            portrait_feats = portrait_dict[uid]
        if uid not in behavior_dict:
            beh_feats = [[0.0]*len(behavior_cols) for _ in range(window_length)]
        else:
            beh_feats = behavior_dict[uid]
        row = df_label.loc[df_label['uid'] == uid].iloc[0]
        out_degree = row['social_out_degree']
        in_degree  = row['social_in_degree']
        churn_val = row['churn_label']
        pay_val   = row['payment_label']
        portrait_list.append(portrait_feats)
        behavior_list.append(beh_feats)
        social_list.append([out_degree, in_degree])
        churn_list.append(churn_val)
        payment_list.append(pay_val)
        user_static_list.append(user_static_dict[uid])

    portrait_ts_tensor = torch.tensor(portrait_list, dtype=torch.float)
    behavior_ts_tensor = torch.tensor(behavior_list, dtype=torch.float)
    social_tensor      = torch.tensor(social_list,   dtype=torch.float)
    churn_label_tensor = torch.tensor(churn_list,    dtype=torch.float)
    payment_label_tensor = torch.tensor(payment_list, dtype=torch.float)

    user_region = [d['region'] for d in user_static_list]
    user_segment = [d['segment'] for d in user_static_list]
    user_region_tensor  = torch.tensor(user_region,  dtype=torch.long)
    user_segment_tensor = torch.tensor(user_segment, dtype=torch.long)
    user_input_dict = {
        'region': user_region_tensor,
        'segment': user_segment_tensor
    }
    return (
        user_input_dict,
        portrait_ts_tensor,
        behavior_ts_tensor,
        social_tensor,
        churn_label_tensor,
        payment_label_tensor
    )

class MyDataset(Dataset):
    def __init__(self,
                 user_input_dict,
                 portrait_ts_tensor,
                 behavior_ts_tensor,
                 social_tensor,
                 churn_label_tensor,
                 payment_label_tensor):
        super().__init__()
        self.user_input_dict = user_input_dict
        self.portrait_ts_tensor = portrait_ts_tensor
        self.behavior_ts_tensor = behavior_ts_tensor
        self.social_tensor = social_tensor
        self.churn_label_tensor = churn_label_tensor
        self.payment_label_tensor = payment_label_tensor
        self.size = self.portrait_ts_tensor.size(0)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        single_user_input = {}
        for k in self.user_input_dict:
            single_user_input[k] = self.user_input_dict[k][idx]
        return (single_user_input,
                self.portrait_ts_tensor[idx],
                self.behavior_ts_tensor[idx],
                self.social_tensor[idx],
                self.churn_label_tensor[idx],
                self.payment_label_tensor[idx])

#############################################
# 2. MODEL DEFINITIONS
#############################################

class CategoricalDenseModel(nn.Module):
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
        embedded = [self.embed_layers[k](x[k]) for k in self.vocab_size_dict.keys()]
        x_cat = torch.cat(embedded, dim=-1)
        return self.mlp(x_cat)

class AutoEncoder(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.encoder(x)

class SkipAutoEncoder(nn.Module):
    def __init__(self, init_channel_dim, depth, output_dim):
        super().__init__()
        self.initial = nn.LazyLinear(init_channel_dim)
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

class WeightedSum(nn.Module):
    def __init__(self, num_inputs=2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
    def forward(self, inputs):
        weights = F.softmax(self.weights, dim=0)
        out = sum(w * inp for w, inp in zip(weights, inputs))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class MultiHeadSelfAttention(nn.Module):
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        return self.fc(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
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

class TransformerEncoder(nn.Module):
    def __init__(self, window_length, feature_dim, dense_layers, trans_output_dim,
                 add_time2vec=True, additional_dropout=False, attention_layer_num=3, n_heads=2):
        super().__init__()
        self.add_time2vec = add_time2vec
        if add_time2vec:
            self.time2vec = nn.Linear(1, feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model=feature_dim, n_heads=n_heads, ff_hidden_dim=feature_dim*2, dropout=0.1)
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
        if self.add_time2vec:
            batch_size, window_length, _ = x.size()
            t = torch.linspace(0, 1, steps=window_length, device=x.device).unsqueeze(0).unsqueeze(-1)
            t = t.expand(batch_size, window_length, 1)
            time_features = self.time2vec(t)
            x = x + time_features
        mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        return self.mlp(x)

class Conv2dUnet(nn.Module):
    def __init__(self, window_length, feature_dim, init_channel_dim=16, depth=2, output_dim=1):
        super().__init__()
        self.depth = depth
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        current_channels = 1
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
        batch_size = x.size(0)
        x = x.unsqueeze(1)  # (batch, 1, window_length, feature_dim)
        downs = []
        for conv in self.down_convs:
            x = conv(x)
            downs.append(x)
            x = F.max_pool2d(x, kernel_size=2, ceil_mode=True)  # ceil_mode to avoid zero dimension
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
    def __init__(self, window_length, portrait_length, output_dim=16):
        super().__init__()
        self.autoencoder = AutoEncoder(output_dim, input_dim=portrait_length)
    def forward(self, x):
        x_last = x[:, -1, :]
        return self.autoencoder(x_last)

# Main multi-task model (CUTAWAS)
class CUTAWAS(nn.Module):
    def __init__(self,
                 vocab_size_dict,    
                 portrait_length,    # dimension per portrait vector
                 behavior_length,    # dimension per behavior vector
                 window_length=5,
                 portrait_ts_out_dim=64,
                 behavior_ts_out_dim=64):
        super().__init__()
        # User branch
        self.user_model = CategoricalDenseModel(vocab_size_dict)
        # Portrait branch
        self.portrait_static_model = PortraitStaticModel(window_length, portrait_length, output_dim=16)
        self.portrait_conv_unet = Conv2dUnet(window_length, portrait_length, init_channel_dim=16, depth=2, output_dim=1)
        self.portrait_transformer_model = TransformerEncoder(
            window_length, portrait_length, dense_layers=[128,64],
            trans_output_dim=portrait_ts_out_dim, add_time2vec=True, additional_dropout=False, attention_layer_num=3, n_heads=2
        )
        self.portrait_weighted_sum = WeightedSum(num_inputs=2)
        # Behavior branch (n_heads=1 due to behavior_length=3)
        self.behavior_conv_unet = Conv2dUnet(window_length, behavior_length, init_channel_dim=16, depth=2, output_dim=1)
        self.behavior_transformer_model = TransformerEncoder(
            window_length, behavior_length, dense_layers=[128,64],
            trans_output_dim=behavior_ts_out_dim, add_time2vec=True, additional_dropout=False, attention_layer_num=3, n_heads=1
        )
        self.behavior_weighted_sum = WeightedSum(num_inputs=2)
        # Social branch
        self.social_embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 16)
        )
        # Final concatenation via SkipAutoEncoder
        total_in_dim = 64 + 16 + portrait_ts_out_dim + behavior_ts_out_dim + 16
        self.skip_autoencoder = SkipAutoEncoder(init_channel_dim=64, depth=2, output_dim=64)
        # Improved churn head: extra hidden layer and dropout
        self.churn_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        # Payment head (unchanged)
        self.payment_head = nn.Linear(64, 1)
    def forward(self, user_input_dict, portrait_input, behavior_input, social_input):
        user_out = self.user_model(user_input_dict)
        portrait_static_out = self.portrait_static_model(portrait_input)
        portrait_conv_out = self.portrait_conv_unet(portrait_input)
        portrait_trans_out = self.portrait_transformer_model(portrait_input)
        portrait_ts_out = self.portrait_weighted_sum([portrait_conv_out, portrait_trans_out])
        behavior_conv_out = self.behavior_conv_unet(behavior_input)
        behavior_trans_out = self.behavior_transformer_model(behavior_input)
        behavior_ts_out = self.behavior_weighted_sum([behavior_conv_out, behavior_trans_out])
        social_emb = self.social_embed(social_input)
        cat = torch.cat([user_out, portrait_static_out, portrait_ts_out, behavior_ts_out, social_emb], dim=-1)
        z = self.skip_autoencoder(cat)
        churn_logit = self.churn_head(z)
        payment_out = self.payment_head(z)
        return churn_logit, payment_out

#############################################
# 3. TRAINING & EVALUATION
#############################################

def train_one_epoch(model, dataloader, optimizer, device, churn_only=True):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        user_input_dict, portrait_ts, behavior_ts, social_ts, churn_label, payment_label = batch
        for k in user_input_dict:
            user_input_dict[k] = user_input_dict[k].to(device)
        portrait_ts = portrait_ts.to(device)
        behavior_ts = behavior_ts.to(device)
        social_ts   = social_ts.to(device)
        churn_label = churn_label.to(device).unsqueeze(-1)
        payment_label = payment_label.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        churn_logit, payment_out = model(user_input_dict, portrait_ts, behavior_ts, social_ts)
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([<some_weight>]).to(device))
        churn_loss = nn.BCEWithLogitsLoss()(churn_logit, churn_label)
        payment_loss = nn.MSELoss()(payment_out, payment_label)
        loss = churn_loss if churn_only else (churn_loss + 0.1 * payment_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, churn_only=True):
    model.eval()
    total_loss = 0.0
    churn_correct = 0
    churn_total = 0
    payment_mse = 0.0
    payment_count = 0
    with torch.no_grad():
        for batch in dataloader:
            user_input_dict, portrait_ts, behavior_ts, social_ts, churn_label, payment_label = batch
            for k in user_input_dict:
                user_input_dict[k] = user_input_dict[k].to(device)
            portrait_ts = portrait_ts.to(device)
            behavior_ts = behavior_ts.to(device)
            social_ts   = social_ts.to(device)
            churn_label = churn_label.to(device).unsqueeze(-1)
            payment_label = payment_label.to(device).unsqueeze(-1)
            churn_logit, payment_out = model(user_input_dict, portrait_ts, behavior_ts, social_ts)
            churn_loss = nn.BCEWithLogitsLoss()(churn_logit, churn_label)
            payment_loss = nn.MSELoss()(payment_out, payment_label)
            loss = churn_loss if churn_only else (churn_loss + 0.1 * payment_loss)
            total_loss += loss.item()
            pred_churn = torch.sigmoid(churn_logit)
            pred_churn_class = (pred_churn > 0.5).float()
            correct = (pred_churn_class == churn_label).sum().item()
            churn_correct += correct
            churn_total += churn_label.size(0)
            diff = payment_out - payment_label
            mse = torch.mean(diff*diff)
            payment_mse += mse.item() * payment_label.size(0)
            payment_count += payment_label.size(0)
    avg_loss = total_loss / len(dataloader)
    churn_acc = churn_correct / churn_total if churn_total > 0 else 0
    payment_mse = payment_mse / payment_count if payment_count > 0 else 0
    return avg_loss, churn_acc, payment_mse

#############################################
# MAIN PIPELINE WITH DIAGNOSTICS
#############################################

if __name__ == "__main__":
    (user_input_dict,
     portrait_ts_tensor,
     behavior_ts_tensor,
     social_tensor,
     churn_label_tensor,
     payment_label_tensor) = load_and_preprocess_data(
                                portrait_csv="sample_data_player_portrait.csv",
                                behavior_csv="sample_data_behavior_sequence.csv",
                                social_csv="sample_data_social_network.csv",
                                label_csv="sample_data_label.csv",
                                window_length=5)
    dataset = MyDataset(user_input_dict,
                        portrait_ts_tensor,
                        behavior_ts_tensor,
                        social_tensor,
                        churn_label_tensor,
                        payment_label_tensor)
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Churn positive ratio: {churn_label_tensor.mean().item():.4f}")
    
    N = len(dataset)
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size_dict = {'region': 5, 'segment': 10}
    portrait_length = portrait_ts_tensor.size(2)
    behavior_length = behavior_ts_tensor.size(2)
    model = CUTAWAS(vocab_size_dict=vocab_size_dict,
                    portrait_length=portrait_length,
                    behavior_length=behavior_length,
                    window_length=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, churn_only=True)
        val_loss, val_churn_acc, _ = evaluate(model, val_loader, device, churn_only=True)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Churn Acc: {val_churn_acc:.4f}")
