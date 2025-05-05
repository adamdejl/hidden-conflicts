import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from einops import einsum
from txai.utils.models import ResidualBlock, Slice


class FtTokenizer(nn.Module):

    def __init__(
        self,
        d_numerical=None,
        feature_cols_ids=None,
        d_token=128,
        bias=True,
    ):
        """
        Initializes the FT-Transformer tokenizer.

        Args:
            d_numerical (int): Number of numerical features. Optional if feature_cols_ids is provided.
            feature_cols_ids (torch.Tensor, optional): IDs of all columns in the input tensor. Defaults to sequential IDs when None.
            d_token (int, optional): Dimensionality of the token embeddings. Defaults to 128.
            bias (bool, optional): Whether to include a bias term for the numerical features. Defaults to True.
        """
        super().__init__()
        if feature_cols_ids is None or (
            feature_cols_ids is not None
            and feature_cols_ids.numel() == feature_cols_ids.unique().numel()
        ):
            if feature_cols_ids is None:
                feature_cols_ids = torch.arange(d_numerical)
            if d_numerical is None:
                d_numerical = len(feature_cols_ids)
            category_idx_offsets = None
            self.category_embeddings = None
        else:
            feature_cols_ids = feature_cols_ids.flatten()
            unique_ids, unique_counts = torch.unique(
                feature_cols_ids, return_counts=True
            )
            numerical_feature_ids = unique_ids[unique_counts == 1]
            d_numerical = len(numerical_feature_ids)
            category_n_options = unique_counts[unique_counts > 1]

            category_idx_offsets = torch.tensor(
                [0] + category_n_options[:-1].detach().tolist()
            ).cumsum(0)
            self.register_buffer("category_idx_offsets", category_idx_offsets)
            self.category_embeddings = nn.Embedding(sum(category_n_options), d_token)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        self.register_buffer("feature_cols_ids", feature_cols_ids)

        # Additional input dimension for the [CLS] token
        self.numerical_weight = nn.Parameter(torch.Tensor(d_numerical + 1, d_token))
        self.numerical_bias = (
            nn.Parameter(torch.Tensor(d_numerical, d_token)) if bias else None
        )
        nn.init.kaiming_uniform_(self.numerical_weight, a=math.sqrt(5))
        if bias:
            nn.init.kaiming_uniform_(self.numerical_bias, a=math.sqrt(5))

    @property
    def numerical_ids(self):
        unique_ids, unique_counts = torch.unique(
            self.feature_cols_ids, return_counts=True
        )
        return unique_ids[unique_counts == 1]

    def get_numerical_bias(self):
        bias = torch.cat(
            [
                torch.zeros(
                    1, self.numerical_bias.shape[1], device=self.numerical_bias.device
                ),
                self.numerical_bias,
            ]
        )
        return bias

    def _get_numerical_embeddings(self, x):
        x_num = x[:, torch.isin(self.feature_cols_ids, self.numerical_ids)]

        # Prepend [CLS] token
        x_num = torch.cat(
            [torch.ones_like(x_num[:, [0]], device=x_num.device), x_num], dim=1
        )
        x_num_emb = einsum(
            x_num, self.numerical_weight, "b n_feat, n_feat d_tok -> b n_feat d_tok"
        )
        if self.numerical_bias is not None:
            bias = self.get_numerical_bias()
            x_num_emb += bias.unsqueeze(0)

        return x_num_emb

    @property
    def categorical_ids(self):
        unique_ids, unique_counts = torch.unique(
            self.feature_cols_ids, return_counts=True
        )
        return unique_ids[unique_counts > 1]

    def _get_categorical_embeddings(self, x):
        if self.category_embeddings is None:
            raise ValueError(
                "Should not be called when there are no categorical features."
            )

        x_cat_emb = []
        for i_cat in range(len(self.categorical_ids)):
            x_cat_one_hot = x[:, self.feature_cols_ids == self.categorical_ids[i_cat]]
            x_cat = x_cat_one_hot.argmax(dim=1)
            feat_emb = self.category_embeddings(
                x_cat + self.category_idx_offsets[i_cat]
            )
            # INFO: Only used in the OOD experiment
            # if torch.isclose(x_cat_one_hot.sum(dim=-1), torch.tensor(0.0)).any():
            #     warnings.warn(
            #         "FtTokenizer attribution: Multiplying embedding with one-hot sum to model missing features."
            #     )
            #     feat_emb *= x_cat_one_hot.sum(dim=-1).unsqueeze(-1)
            x_cat_emb.append(feat_emb)
        return torch.stack(x_cat_emb, dim=1)

    @property
    def n_feat(self):
        return len(self.numerical_weight) + (
            0 if self.category_idx_offsets is None else len(self.category_idx_offsets)
        )

    def forward(self, x):
        x_emb = self._get_numerical_embeddings(x)
        if self.category_embeddings is not None:
            x_cat_emb = self._get_categorical_embeddings(x)
            x_emb = torch.cat([x_emb, x_cat_emb], dim=1)
        return x_emb


class FtMultiheadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=8, dropout=0.2):
        if n_heads > 1:
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_head = d_model // n_heads

        super().__init__()
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, d_head))
        self.W_O = nn.Parameter(torch.empty(n_heads, d_head, d_model))
        self.b_Q = nn.Parameter(torch.empty(n_heads, d_head))
        self.b_K = nn.Parameter(torch.empty(n_heads, d_head))
        self.b_V = nn.Parameter(torch.empty(n_heads, d_head))
        self.b_O = nn.Parameter(torch.empty(d_model))
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        for m in [self.W_K, self.W_Q, self.W_V]:
            nn.init.xavier_uniform_(m, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_O)
        for b in [self.b_Q, self.b_K, self.b_V, self.b_O]:
            nn.init.zeros_(b)

    def compute_attention(self, x):
        q = (
            einsum(
                x,
                self.W_Q,
                "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
            )
            + self.b_Q
        )
        k = (
            einsum(
                x,
                self.W_K,
                "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
            )
            + self.b_K
        )

        attn_scores = einsum(
            q,
            k,
            "batch n_feat_q n_heads d_head, batch n_feat_k n_heads d_head -> batch n_heads n_feat_q n_feat_k",
        )
        attn_scores = attn_scores / self.d_head**0.5
        attn_pattern = F.softmax(attn_scores, dim=-1)

        if self.dropout is not None:
            attn_pattern = self.dropout(attn_pattern)

        return attn_pattern

    def forward(self, x):
        attn_pattern = self.compute_attention(x)

        v = (
            einsum(
                x,
                self.W_V,
                "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
            )
            + self.b_V
        )

        v_out = einsum(
            v,
            attn_pattern,
            "batch n_feat_k n_heads d_head, batch n_heads n_feat_q n_feat_k -> batch n_feat_q n_heads d_head",
        )

        attn_out = (
            einsum(
                v_out,
                self.W_O,
                "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
            )
            + self.b_O
        )

        return attn_out


def construct_ft_transformer(
    *,
    d_numerical=None,
    feature_cols_ids=None,
    d_model=128,
    token_bias=True,
    n_layers=3,
    n_heads=8,
    d_ffn_factor=4 / 3,
    attention_dropout=0.2,
    ffn_dropout=0.1,
    seed=None,
    d_out,
    use_relu=False,
):
    if seed is not None:
        torch.manual_seed(seed)
    tokenizer = FtTokenizer(d_numerical, feature_cols_ids, d_model, token_bias)
    layers = [tokenizer]
    for i in range(n_layers):
        internal_layers = []
        if i > 0:
            # FT-Transformer authors advise against normalising in the first layer
            internal_layers.append(nn.LayerNorm(d_model))
        internal_layers.extend(
            [
                FtMultiheadAttention(d_model, n_heads, attention_dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, int(d_model * d_ffn_factor)),
                nn.GELU() if not use_relu else nn.ReLU(),
            ]
        )
        if ffn_dropout > 0:
            internal_layers.append(nn.Dropout(ffn_dropout))
        internal_layers.append(nn.Linear(int(d_model * d_ffn_factor), d_model))
        block = nn.Sequential(*internal_layers)
        layers.append(ResidualBlock(nn.Identity(), block))
    # Output head
    layers.extend(
        [
            Slice((slice(None), -1)),
            nn.GELU() if not use_relu else nn.ReLU(),
            nn.Linear(d_model, d_out),
        ]
    )
    return nn.Sequential(*layers)
