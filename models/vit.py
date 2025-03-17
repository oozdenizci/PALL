import math
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from models.base import *

__all__ = ['VisionTransformer', 'vit_t_8', 'vit_t_16']


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V):
        """
            inputs:
                Q: (batch_size, n_head, seq_len1, d_k)
                K: (batch_size, n_head, seq_len2, d_k)
                V: (batch_size, n_head, seq_len2, d_v)
                mask: (batch_size, n_heads, seq_len1, seq_len2)
            outputs:
                context: (batch_size, n_heads, seq_len1, d_v)
                attn: (batch_size, n_heads, seq_len1, seq_len2)
        """
        # scores (batch_size, n_heads, seq_len1, seq_len2)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(Q.shape[-1])
        # attn (batch_size, n_heads, seq_len1, seq_len2)
        attn = self.dropout(F.softmax(scores, dim=-1))
        # context (batch_size, n_heads, seq_len1, d_v)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k=None, d_v=None, p_dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k if d_k is not None else d_model // n_head
        self.d_v = d_v if d_v is not None else d_model // n_head
        self.W_Q = nn.Linear(d_model, self.d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * n_head, bias=False)
        self.sdpa = ScaledDotProductAttention(p_dropout)
        self.fc = nn.Linear(self.d_v * n_head, d_model, bias=False)

    def forward(self, Q, K, V):
        """
            inputs:
                Q: (batch_size, seq_len1, d_model)
                K: (batch_size, seq_len2, d_model)
                V: (batch_size, seq_len2, d_model]
                mask: (batch_size, seq_len1, seq_len2)
            outputs:
                output: (batch_size, seq_len1, d_model)
                attn: (batch_size, n_heads, seq_len1, seq_len2)
        """
        batch_size = Q.shape[0]
        # Q -> (batch_size, seq_len1, d_k * n_head) -> (batch_size, seq_len1, n_head, d_k) -> (batch_size, n_head, seq_len1, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # K -> (batch_size, seq_len2, d_k * n_head) -> (batch_size, seq_len2, n_head, d_k) -> (batch_size, n_head, seq_len2, d_k)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # V -> (batch_size, seq_len2, d_v * n_head) -> (batch_size, seq_len2, n_head, d_v) -> (batch_size, n_head, seq_len2, d_v)
        V = self.W_V(V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        # mask -> (batch_size, 1, seq_len1, seq_len2) -> (batch_size, n_heads, seq_len1, seq_len2)
        context, attn = self.sdpa(Q, K, V)

        # (batch_size, seq_len1, n_heads * d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)

        return output, attn


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_params=True):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)
        self.mha = MultiHeadAttention(hidden_dim, num_heads, p_dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)
        self.mlp = nn.Sequential(torch.nn.Linear(hidden_dim, mlp_dim, bias=True),
                                 torch.nn.GELU(),
                                 torch.nn.Dropout(dropout),
                                 torch.nn.Linear(mlp_dim, hidden_dim, bias=True),
                                 torch.nn.Dropout(dropout))

    def forward(self, input):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.mha(x, x, x)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class VisionTransformer(BaseModel):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1,
                 attention_dropout=0.1, num_classes=200, norm_params=True):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_params = norm_params

        seq_length = (image_size // patch_size) ** 2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=True)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1
        self.seq_length = seq_length

        # Transformer Encoder
        # Note that batch_size is on the first dim because we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.d_out = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layers_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_params,
            )
        self.layers = nn.Sequential(layers)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=norm_params, bias=norm_params)

        self.classifier = nn.Linear(hidden_dim, num_classes, bias=False)

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv1(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, returnt='out'):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Forward pass from the encoder
        x = x + self.pos_embedding
        x = self.ln(self.layers(self.d_out(x)))

        # Classifier "token" as used by standard language architectures
        feature = x[:, 0]

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return out, feature

        raise NotImplementedError("Unknown return type")


def vit_t_8(num_classes, norm_params=False, n_tasks=1, sparsity=None):
    return VisionTransformer(image_size=64, patch_size=8, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768,
                             num_classes=num_classes, norm_params=norm_params)


def vit_t_16(num_classes, norm_params=False, n_tasks=1, sparsity=None):
    return VisionTransformer(image_size=64, patch_size=16, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768,
                             num_classes=num_classes, norm_params=norm_params)
