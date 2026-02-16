import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from einops import repeat


def modulate(x, shift, scale):
    """Apply shift and scale modulation: y = (x + shift) * (1 + scale)"""
    return (x + shift) * (1 + scale)


class TimeConditionedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Transformer encoder layer with time conditioning via Adaptive Layer Normalization (AdaLN).

    Args:
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        activation: Activation function
        batch_first: Whether batch is first dimension
        time_dim: Optional dimension of time conditioning. If None, assumes time_cond has shape (batch, d_model)
    """

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            batch_first: bool = True,
            time_dim: Optional[int] = None,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
        )

        self.d_model = d_model
        time_dim = time_dim or d_model
        self.batch_first = batch_first
        # AdaLN modulation network
        # Outputs: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model, bias=True),
        )

        # Initialize AdaLN parameters
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        bias = torch.ones(6 * d_model)
        bias[d_model: 2 * d_model] = 1  # scale_msa default to 1
        bias[5 * d_model:] = 1  # gate_mlp default to 1
        self.adaLN_modulation[-1].bias = nn.Parameter(bias)

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            time_cond: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence of shape (batch, seq_len, d_model) or (seq_len, batch, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
            time_cond: Time conditioning of shape (batch, time_dim). Required for time conditioning.
            output_attentions: Whether to return attention weights

        Returns:
            Output tensor of same shape as src, or tuple (output, attention_weights) if output_attentions=True
        """

        if time_cond is None:
            # Fall back to standard transformer behavior
            return super().forward(
                src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )

        # Get modulation parameters from time conditioning
        # Shape: (batch, 6 * d_model)
        modulation = self.adaLN_modulation(time_cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(
            6, dim=1
        )

        # Handle batch_first conversion if needed
        if self.batch_first:
            # src shape: (batch, seq_len, d_model)
            # Add sequence dimension to modulation params for broadcasting
            shift_msa = shift_msa.unsqueeze(1)  # (batch, 1, d_model)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        else:
            # src shape: (seq_len, batch, d_model)
            # Add sequence dimension at the beginning
            shift_msa = shift_msa.unsqueeze(0)  # (1, batch, d_model)
            scale_msa = scale_msa.unsqueeze(0)
            gate_msa = gate_msa.unsqueeze(0)
            shift_mlp = shift_mlp.unsqueeze(0)
            scale_mlp = scale_mlp.unsqueeze(0)
            gate_mlp = gate_mlp.unsqueeze(0)

        # Self-attention block with AdaLN
        residual = src
        x = self.norm1(src)
        x = modulate(x, shift_msa, scale_msa)
        x = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=output_attentions,
        )

        x, attn_weights = x

        x = x * gate_msa
        x = residual + x

        # Feedforward block with AdaLN
        residual = x
        y = self.norm2(x)
        y = modulate(y, shift_mlp, scale_mlp)
        y = self.linear2(self.dropout(self.activation(self.linear1(y))))
        y = y * gate_mlp
        x = residual + y

        if output_attentions:
            return x, attn_weights
        return x


class TimeConditionedTransformerEncoder(nn.TransformerEncoder):
    """
    Transformer encoder stack with time conditioning support.

    Usage:
        encoder_layer = TimeConditionedTransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        encoder = TimeConditionedTransformerEncoder(encoder_layer, num_layers=num_layers)

        output = encoder(src, time_cond=time_cond)
    """

    def forward(
            self,
            src: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            time_cond: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence
            mask: Attention mask
            src_key_padding_mask: Padding mask
            time_cond: Time conditioning of shape (batch, time_dim)
            output_attentions: Whether to return attention weights from all layers

        Returns:
            Output tensor or tuple of (output, attention_weights_list)
        """

        output = src
        attentions = [] if output_attentions else None

        for mod in self.layers:
            if output_attentions:
                output, attn = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    time_cond=time_cond,
                    output_attentions=True,
                )
                attentions.append(attn)
            else:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    time_cond=time_cond,
                    output_attentions=False,
                )

        if self.norm is not None:
            output = self.norm(output)

        if output_attentions:
            return output, attentions
        return output


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)



class ACTORStyleEncoder(nn.Module):
    _timestep_warning_shown = False

    # Similar to ACTOR but "action agnostic" and more general
    def __init__(
        self,
        nfeats: int,
        vae: bool,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        text=False,
    ) -> None:
        super().__init__()

        self.nfeats = nfeats
        self.projection = nn.Linear(nfeats, latent_dim)

        self.vae = vae
        self.nbtokens = 2 if vae else 1
        self.tokens = nn.Parameter(torch.randn(self.nbtokens, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout=dropout, batch_first=True
        )
        self.text = text
        if self.text:
            seq_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )

            self.seqTransEncoder = nn.TransformerEncoder(
                seq_trans_encoder_layer, num_layers=num_layers
            )

        else:
            seq_trans_encoder_layer = TimeConditionedTransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )

            self.seqTransEncoder = TimeConditionedTransformerEncoder(
                seq_trans_encoder_layer, num_layers=num_layers
            )
            self.timestepEmbedder = TimestepEmbedder(latent_dim)


    def forward(self, x_dict: Dict) -> Tensor:
        x = x_dict["x"]
        mask = x_dict["mask"]

        x = self.projection(x)

        device = x.device
        bs = len(x)

        tokens = repeat(self.tokens, "nbtoken dim -> bs nbtoken dim", bs=bs)
        xseq = torch.cat((tokens, x), 1)

        token_mask = torch.ones((bs, self.nbtokens), dtype=bool, device=device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)

        if self.text:
            final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        else:
            # utility print
            if not self.__class__._timestep_warning_shown:
                if 't' in x_dict:
                    print("✅  - timestep provided to ACTORStyleEncoder")
                else:
                    print("⚠️  - Warning: no timestep provided to ACTORStyleEncoder, using zeros")
                self.__class__._timestep_warning_shown = True

            if 't' in x_dict:
                t = x_dict["t"].to(device)
            else:
                t = torch.zeros(x_dict["x"].shape[0], device=x_dict["x"].device)
            t = self.timestepEmbedder(t)

            final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask, time_cond=t)

        return final[:, : self.nbtokens]


class ACTORStyleDecoder(nn.Module):
    # Similar to ACTOR Decoder

    def __init__(
        self,
        nfeats: int,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        output_feats = nfeats
        self.nfeats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.seqTransDecoder = nn.TransformerDecoder(
            seq_trans_decoder_layer, num_layers=num_layers
        )

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z_dict: Dict) -> Tensor:
        z = z_dict["z"]
        mask = z_dict["mask"]

        latent_dim = z.shape[1]
        bs, nframes = mask.shape

        z = z[:, None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(bs, nframes, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(
            tgt=time_queries, memory=z, tgt_key_padding_mask=~mask
        )

        output = self.final_layer(output)
        # zero for padded area
        output[~mask] = 0
        return output