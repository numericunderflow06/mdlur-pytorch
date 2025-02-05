import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    CategoricalDenseModel,
    WeightedSum,
    AutoEncoder,
    SkipAutoEncoder,
    Conv2dUnet,
    TransformerEncoder,
)

class CUTAWAS(nn.Module):
    def __init__(self, behavior_length, portrait_length, window_length, target_day, vocab_size_dict, **kwargs):
        """
        PyTorch implementation of the CUTAWAS model.

        Args:
            behavior_length (int): Length of behavior features.
            portrait_length (int): Length of portrait features.
            window_length (int): Number of time steps.
            target_day (int): Output dimensionality.
            vocab_size_dict (dict).
        """
        super(CUTAWAS, self).__init__()
        
        self.behavior_length = behavior_length
        self.portrait_length = portrait_length
        self.window_length = window_length
        self.target_day = target_day
        self.vocab_size_dict = vocab_size_dict
        self.dense_layers = [128, 64]
        self.attention_num = 3
        self.trans_output_dim = 64

        # Build the user model.
        self.user_model = self.build_user_model()

        # Build the portrait static submodel
        self.portrait_static_autoencoder = AutoEncoder(16)

        # Build the portrait time-series submodels
        self.portrait_conv_unet = self.get_conv_unet(self.portrait_length)
        self.portrait_transformer = self.get_transformer_encoder(self.portrait_length)
        # Use separate instances of WeightedSum for portrait and behavior streams.
        self.weighted_sum_portrait = WeightedSum()

        # Build the behavior time-series submodels
        self.behavior_conv_unet = self.get_conv_unet(self.behavior_length)
        self.behavior_transformer = self.get_transformer_encoder(self.behavior_length)
        self.weighted_sum_behavior = WeightedSum()

        # Build the final skip-autoencoder (the dense layers)
        self.skip_autoencoder = SkipAutoEncoder(init_channel_dim=32, depth=2, output_dim=self.target_day)

    def build_user_model(self):
        # In TensorFlow, the user model was:
        #   CategoricalDenseModel()(self.vocab_size_dict)
        return CategoricalDenseModel(self.vocab_size_dict)

    def get_transformer_encoder(self, feature_dim):
        return TransformerEncoder(
            window_length=self.window_length,
            feature_dim=feature_dim,
            dense_layers=self.dense_layers,
            trans_output_dim=self.trans_output_dim,
            add_time2vec=True,
            additional_dropout=False,
            attention_layer_num=self.attention_num,
        )

    def get_conv_unet(self, feature_dim):
        return Conv2dUnet(
            window_length=self.window_length,
            feature_dim=feature_dim,
            init_channel_dim=16,
            depth=2,
            output_dim=1
        )

    def forward(self, user_inputs, portrait, behavior):
        """
        Args:
            user_inputs: Input(s) for the user model
            portrait: Tensor of shape [batch_size, window_length, portrait_length]
            behavior: Tensor of shape [batch_size, window_length, behavior_length]
        Returns:
            Tensor of shape [batch_size, target_day]
        """
        # Process user features.
        user_out = self.user_model(user_inputs)

        # Process portrait static features.
        # Take the last time step from the portrait input.
        portrait_static_input = portrait[:, -1, :]  # shape: [batch, portrait_length]
        portrait_static_out = self.portrait_static_autoencoder(portrait_static_input)

        # Process portrait time-series features.
        portrait_conv_out = self.portrait_conv_unet(portrait)
        portrait_transformer_out = self.portrait_transformer(portrait)
        # Weighted ensemble of the two portrait outputs.
        portrait_ts_out = self.weighted_sum_portrait([portrait_conv_out, portrait_transformer_out])

        # Process behavior time-series features.
        behavior_conv_out = self.behavior_conv_unet(behavior)
        behavior_transformer_out = self.behavior_transformer(behavior)
        behavior_ts_out = self.weighted_sum_behavior([behavior_conv_out, behavior_transformer_out])

        # Concatenate all features along the last dimension.
        concatenated = torch.cat(
            [user_out, portrait_static_out, portrait_ts_out, behavior_ts_out],
            dim=-1
        )

        # Pass through the final skip-autoencoder/dense layers.
        output = self.skip_autoencoder(concatenated)
        return output

    def get_config(self):
        return {
            'behavior_length': self.behavior_length,
            'portrait_length': self.portrait_length,
            'window_length': self.window_length,
            'target_day': self.target_day,
            'vocab_size_dict': self.vocab_size_dict,
            'dense_layers': self.dense_layers,
            'attention_num': self.attention_num,
            'trans_output_dim': self.trans_output_dim,
        }
