import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from copy import deepcopy
from einops import einsum
from enum import auto, Enum
from types import MethodType

from txai.utils import OpenXaiAnn
from txai.utils.models import ResidualBlock, Slice
from txai.models import FtTokenizer, FtMultiheadAttention


class CafeNgExplainer:
    def __init__(self, model, c=0.5, epsilon=1e-9, report_discrepancy=False):
        self.model = model
        self.converters = {
            nn.Sequential: self.convert_sequential,
            # FIXME: Hotfix for OpenXAI models, should check
            #        parent class instead.
            OpenXaiAnn: self.convert_sequential,
            nn.Linear: self.convert_linear,
            nn.ReLU: self.convert_activation,
            nn.GELU: self.convert_activation,
            nn.Sigmoid: self.convert_activation,
            nn.Tanh: self.convert_activation,
            nn.Conv2d: self.convert_conv2d,
            nn.Flatten: self.convert_flatten,
            nn.Identity: self.convert_identity,
            nn.Dropout: self.convert_identity,
            nn.MaxPool2d: self.convert_maxpool2d,
            nn.AvgPool2d: self.convert_avgpool2d,
            nn.AdaptiveAvgPool2d: self.convert_avgpool2d,
            FtTokenizer: self.convert_ft_tokenizer,
            FtMultiheadAttention: self.convert_ft_attention,
            nn.BatchNorm2d: self.convert_batchnorm2d,
            nn.LayerNorm: self.convert_layernorm,
            nn.Embedding: self.convert_error,
            ResidualBlock: self.convert_residual,
            Slice: self.convert_slice,
        }
        self.c = c
        self.epsilon = epsilon
        self.report_discrepancy = report_discrepancy

        self.model.apply(self.convert_module)

    def convert_module(self, module):
        """
        Converts the supplied module, adding the `attribute` method.
        """
        converter = self.converters.get(type(module))
        if converter:
            converter(module)
        else:
            raise ValueError(
                f"Attribution not implemented for module type {module.__class__.__name__}"
            )

    @staticmethod
    def add_attr_fun(module, method):
        """
        Adds the specified method as the `attribute_forward` method for the supplied module.
        """
        module.attribute_forward = MethodType(method, module)

    def convert_sequential(self, module):
        report_discrepancy = self.report_discrepancy

        def attribute_sequential(self, ref_as, s_plus, s_minus, deep=False):
            """
            Propagates attribution scores through a sequential model.

            Delegates the computation of all rules to submodules.
            """
            for i, module in enumerate(self):
                prev_ref_as, prev_s_plus, prev_s_minus = ref_as, s_plus, s_minus

                ref_as, s_plus, s_minus = module.attribute_forward(
                    ref_as, s_plus, s_minus, deep=deep
                )

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_sequential)

    def convert_linear(self, module):
        def attribute_linear(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a linear layer.

            Directly implements the linear rule.
            """
            if self.bias is not None:
                if prev_s_plus.ndim == 3:
                    # Linear layer in a transformer/sequential model
                    self.expanded_bias = self.bias.repeat(len(prev_s_plus)).reshape(
                        len(prev_s_plus), 1, -1
                    )
                else:
                    if prev_s_plus.ndim != 2:
                        warnings.warn(
                            "Unexpected input shape for linear layer — check results"
                        )
                    self.expanded_bias = self.bias.repeat(len(prev_s_plus)).reshape(
                        len(prev_s_plus), -1
                    )
                self.expanded_bias.retain_grad()
            else:
                self.expanded_bias = None

            if deep:
                self.prev_s_plus = prev_s_plus
                self.prev_s_plus.retain_grad()
                self.prev_s_minus = prev_s_minus
                self.prev_s_minus.retain_grad()

            s_plus = (
                prev_s_plus @ F.relu(self.weight.T)
                + prev_s_minus @ F.relu(-self.weight.T)
                + F.relu(
                    self.expanded_bias
                    if self.expanded_bias is not None
                    else torch.tensor(0.0).to(ref_as.device)
                )
            )
            s_minus = (
                prev_s_plus @ F.relu(-self.weight.T)
                + prev_s_minus @ F.relu(self.weight.T)
                + F.relu(
                    -self.expanded_bias
                    if self.expanded_bias is not None
                    else torch.tensor(0.0).to(ref_as.device)
                )
            )

            ref_as = self(ref_as) - (
                self.bias
                if self.bias is not None
                else torch.tensor(0.0).to(ref_as.device)
            )

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_linear)

    def convert_activation(self, module):
        c = self.c
        epsilon = self.epsilon

        def attribute_activation(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a non-linear activation layer.

            Directly implements the activation rule.
            """
            if deep:
                self.prev_s_plus = prev_s_plus
                self.prev_s_plus.retain_grad()
                self.prev_s_minus = prev_s_minus
                self.prev_s_minus.retain_grad()

            prev_s_star = prev_s_plus - prev_s_minus

            # Binary matrices for the conditions e_star >= 0 and e_star < 0
            ind_pos = prev_s_star >= 0
            ind_neg = prev_s_star < 0
            sign = torch.sign(prev_s_star)

            # Compute total effects (combination of effects from reference and input)
            t_plus = prev_s_plus + ref_as
            t_minus = -prev_s_minus + ref_as
            t_star = prev_s_star + ref_as

            p_plus_plus = (
                # Observed positive impact of combined features that are positive
                ind_pos * F.relu(self(t_star) - self(ref_as))
                # Additional hypothetical positive impact of cancelled features
                + torch.maximum(
                    F.relu(self(t_plus) - self(ind_pos * t_star + ind_neg * ref_as)),
                    F.relu(self(ind_pos * ref_as + ind_neg * t_star) - self(t_minus)),
                )
            )
            p_plus_minus = (
                # Observed negative impact of combined features that are positive
                ind_pos * F.relu(self(ref_as) - self(t_star))
                # Additional hypothetical negative impact of cancelled features
                + torch.maximum(
                    F.relu(self(ind_pos * t_star + ind_neg * ref_as) - self(t_plus)),
                    F.relu(self(t_minus) - self(ind_pos * ref_as + ind_neg * t_star)),
                )
            )
            p_minus_plus = (
                # Observed positive impact of combined features that are negative
                ind_neg * F.relu(self(t_star) - self(ref_as))
                # Additional hypothetical positive impact of cancelled features
                + torch.maximum(
                    F.relu(self(ind_pos * t_star + ind_neg * ref_as) - self(t_plus)),
                    F.relu(self(t_minus) - self(ind_pos * ref_as + ind_neg * t_star)),
                )
            )
            p_minus_minus = (
                # Observed negative impact of combined features that are negative
                ind_neg * F.relu(self(ref_as) - self(t_star))
                # Additional hypothetical negative impact of cancelled features
                + torch.maximum(
                    F.relu(self(t_plus) - self(ind_pos * t_star + ind_neg * ref_as)),
                    F.relu(self(ind_pos * ref_as + ind_neg * t_star) - self(t_minus)),
                )
            )

            l_plus_plus = torch.minimum(
                prev_s_plus
                * F.relu(sign * (self(t_star) - self(ref_as)))
                / (prev_s_star.abs() + epsilon),
                p_plus_plus,
            )
            l_plus_minus = torch.minimum(
                prev_s_plus
                * F.relu(sign * (self(ref_as) - self(t_star)))
                / (prev_s_star.abs() + epsilon),
                p_plus_minus,
            )
            l_minus_plus = torch.minimum(
                prev_s_minus
                * F.relu(sign * (self(ref_as) - self(t_star)))
                / (prev_s_star.abs() + epsilon),
                p_minus_plus,
            )
            l_minus_minus = torch.minimum(
                prev_s_minus
                * F.relu(sign * (self(t_star) - self(ref_as)))
                / (prev_s_star.abs() + epsilon),
                p_minus_minus,
            )

            # Compute the attribution multipliers
            # Detaching the multipliers ensures that they are treated as linear weights during the
            # backward pass
            m_plus_plus = (
                ((1 - c) * l_plus_plus + c * p_plus_plus) / (prev_s_plus + epsilon)
            ).detach()
            m_minus_plus = (
                ((1 - c) * l_minus_plus + c * p_minus_plus) / (prev_s_minus + epsilon)
            ).detach()
            m_plus_minus = (
                ((1 - c) * l_plus_minus + c * p_plus_minus) / (prev_s_plus + epsilon)
            ).detach()
            m_minus_minus = (
                ((1 - c) * l_minus_minus + c * p_minus_minus) / (prev_s_minus + epsilon)
            ).detach()

            # Compute the new attribution scores
            s_plus = m_plus_plus * prev_s_plus + m_minus_plus * prev_s_minus
            s_minus = m_plus_minus * prev_s_plus + m_minus_minus * prev_s_minus
            ref_as = self(ref_as)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_activation)

    def convert_conv2d(self, module):
        def attribute_conv2d(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attributions through a 2D convolutional layer.
            """
            # TODO: Check and ensure CNN support for deep attributions
            if deep:
                warnings.warn(
                    "Conv2d: Deep attributions have not yet been implemented for CNNs"
                )

            if hasattr(self, "expanded_bias"):
                del self.expanded_bias

            conv_plus = deepcopy(self)
            conv_plus.weight = nn.Parameter(F.relu(conv_plus.weight))
            conv_plus.bias = None
            conv_minus = deepcopy(self)
            conv_minus.weight = nn.Parameter(F.relu(-conv_minus.weight))
            conv_minus.bias = None
            conv_no_bias = deepcopy(self)
            conv_no_bias.bias = None

            if self.bias is not None:
                self.expanded_bias = self.bias.repeat(len(prev_s_plus)).reshape(
                    len(prev_s_plus), *self.bias.shape
                )
                self.expanded_bias.retain_grad()
            else:
                self.expanded_bias = None

            s_plus = (
                conv_plus(prev_s_plus)
                + conv_minus(prev_s_minus)
                + F.relu(
                    self.expanded_bias.unsqueeze(-1).unsqueeze(-1)
                    if self.expanded_bias is not None
                    else torch.tensor(0.0).to(ref_as.device)
                )
            )
            s_minus = (
                conv_minus(prev_s_plus)
                + conv_plus(prev_s_minus)
                + F.relu(
                    -self.expanded_bias.unsqueeze(-1).unsqueeze(-1)
                    if self.expanded_bias is not None
                    else torch.tensor(0.0).to(ref_as.device)
                )
            )
            ref_as = conv_no_bias(ref_as)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_conv2d)

    def convert_flatten(self, module):
        def attribute_flatten(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a flatten module.
            """
            s_plus = self(prev_s_plus)
            s_minus = self(prev_s_minus)
            ref_as = self(ref_as)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_flatten)

    def convert_identity(self, module):
        def attribute_identity(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a dropout/identity module.
            """
            if self.training:
                raise ValueError(
                    f"Computing attributions while training is not supported ({self})"
                )

            return ref_as, prev_s_plus, prev_s_minus

        self.add_attr_fun(module, attribute_identity)

    def convert_maxpool2d(self, module):
        epsilon = self.epsilon

        def attribute_maxpool2d(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a 2D max pooling module.
            """
            # TODO: Check and ensure CNN support for deep attributions
            if deep:
                warnings.warn(
                    "MaxPool2d: Deep attributions have not yet been implemented for CNNs"
                )

            prev_s_star = prev_s_plus - prev_s_minus
            t_star = ref_as + prev_s_star
            maxpool_copy = deepcopy(self)
            maxpool_copy.return_indices = True

            # First, compute the maxpool operation and the corresponding indices
            # Note that the indices are per sample and channel, so we only flatten
            # the spatial dimensions.
            ref_maxima, ref_indices = maxpool_copy(ref_as)
            ref_indices = ref_indices.flatten(start_dim=2)
            curr_maxima, curr_indices = maxpool_copy(t_star)
            curr_indices = curr_indices.flatten(start_dim=2)

            # For elements where the maximum is the same, directly propagate the scores
            # We use gather to collect the elements selected by the max pooling.
            # Note that we cannot use max pool directly to get these elements,
            # as we are opearating on positive/negative scores rather than the
            # underlying input.
            same_indices_mask = torch.zeros_like(ref_indices)
            same_indices_mask[ref_indices == curr_indices] = 1
            s_plus_candidates = prev_s_plus.flatten(start_dim=2).gather(
                -1, curr_indices
            )
            s_plus_same = torch.where(same_indices_mask.bool(), s_plus_candidates, 0)
            s_minus_candidates = prev_s_minus.flatten(start_dim=2).gather(
                -1, curr_indices
            )
            s_minus_same = torch.where(same_indices_mask.bool(), s_minus_candidates, 0)

            # Where the maximum is different, interpolate the scores for the reference
            # and the current input
            new_maxima_ref = ref_as.flatten(start_dim=2).gather(-1, curr_indices)
            new_maxima_curr = t_star.flatten(start_dim=2).gather(-1, curr_indices)
            ref_maxima_ref = ref_as.flatten(start_dim=2).gather(-1, ref_indices)
            ref_maxima_curr = t_star.flatten(start_dim=2).gather(-1, ref_indices)
            new_maxima_deltas = new_maxima_curr - new_maxima_ref
            ref_maxima_deltas = ref_maxima_curr - ref_maxima_ref
            # Intersections indicates input values for which the max values from the reference
            # and the actual input are equal.
            intersections = (
                new_maxima_deltas * ref_maxima_ref - ref_maxima_deltas * new_maxima_ref
            ) / (new_maxima_deltas - ref_maxima_deltas + epsilon)
            # Using these intersections, we can compute the respective influences of the changes
            # in the reference maximum and the current input maximum compared to the reference.
            ref_multipliers = (
                (ref_maxima_ref - intersections)
                / (ref_maxima_ref - ref_maxima_curr + epsilon)
            ).detach()
            new_multipliers = (
                (new_maxima_curr - intersections)
                / (new_maxima_curr - new_maxima_ref + epsilon)
            ).detach()
            s_plus_candidates = (
                ref_multipliers
                * prev_s_plus.flatten(start_dim=2).gather(-1, ref_indices)
            ) + (
                new_multipliers
                * prev_s_plus.flatten(start_dim=2).gather(-1, curr_indices)
            )
            s_plus_diff = torch.where(~same_indices_mask.bool(), s_plus_candidates, 0)
            s_minus_candidates = (
                ref_multipliers
                * prev_s_minus.flatten(start_dim=2).gather(-1, ref_indices)
            ) + (
                new_multipliers
                * prev_s_minus.flatten(start_dim=2).gather(-1, curr_indices)
            )
            s_minus_diff = torch.where(~same_indices_mask.bool(), s_minus_candidates, 0)

            # Join the scores for identical and different maxima
            s_plus = s_plus_same + s_plus_diff
            s_minus = s_minus_same + s_minus_diff

            # Reshape scores to the correct shape
            s_plus = s_plus.reshape_as(curr_maxima)
            s_minus = s_minus.reshape_as(curr_maxima)

            ref_as = self(ref_as)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_maxpool2d)

    def convert_avgpool2d(self, module):
        def attribute_avgpool2d(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a 2D average pooling module
            """
            if deep:
                warnings.warn(
                    "AvgPool2d: Deep attributions have not yet been implemented for CNNs"
                )

            ref_as = self(ref_as)
            s_plus = self(prev_s_plus)
            s_minus = self(prev_s_minus)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_avgpool2d)

    def convert_ft_tokenizer(self, module):
        def attribute_ft_tokenizer(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a FT-Transformer tokenizer.
            """
            if deep:
                warnings.warn(
                    "FtTokenizer: Deep attributions have not yet been implemented for transformers"
                )

            self.expanded_bias = []

            def get_num_cols(x, zero_cls=False):
                x_num = x[:, torch.isin(self.feature_cols_ids, self.numerical_ids)]

                # We interpret the [CLS] embedding as another bias term
                if not self.expanded_bias:
                    self.expanded_bias.append(
                        torch.ones_like(x_num[:, [0]], device=x_num.device)
                    )
                    self.expanded_bias[0].requires_grad = True

                # [CLS] "bias" should only be included in the S+ term
                # (possibly being flipped to S- by applying numerical_weight)
                x_num = torch.cat(
                    [
                        (0 if zero_cls else 1) * self.expanded_bias[0],
                        x_num,
                    ],
                    dim=1,
                )

                return x_num

            ref_as_num = get_num_cols(ref_as, zero_cls=True)
            ref_as_emb = einsum(
                ref_as_num,
                self.numerical_weight,
                "b n_feat, n_feat d_tok -> b n_feat d_tok",
            )

            s_plus_num = get_num_cols(prev_s_plus)
            s_minus_num = get_num_cols(prev_s_minus, zero_cls=True)

            if self.numerical_bias is not None:
                self.expanded_bias.append(
                    self.get_numerical_bias()
                    .repeat(len(prev_s_plus), 1)
                    .reshape(len(prev_s_plus), *self.get_numerical_bias().shape)
                    .detach()
                )
                self.expanded_bias[-1].requires_grad = True

            s_plus_emb = (
                einsum(
                    s_plus_num,
                    F.relu(self.numerical_weight),
                    "b n_feat, n_feat d_tok -> b n_feat d_tok",
                )
                + einsum(
                    s_minus_num,
                    F.relu(-self.numerical_weight),
                    "b n_feat, n_feat d_tok -> b n_feat d_tok",
                )
                + (
                    F.relu(self.expanded_bias[1])
                    if self.numerical_bias is not None
                    else torch.tensor(0.0)
                )
            )
            s_minus_emb = (
                einsum(
                    s_plus_num,
                    F.relu(-self.numerical_weight),
                    "b n_feat, n_feat d_tok -> b n_feat d_tok",
                )
                + einsum(
                    s_minus_num,
                    F.relu(self.numerical_weight),
                    "b n_feat, n_feat d_tok -> b n_feat d_tok",
                )
                + (
                    F.relu(-self.expanded_bias[1])
                    if self.numerical_bias is not None
                    else torch.tensor(0.0)
                )
            )

            if self.category_embeddings is not None:
                ref_as_cat_emb = []
                s_plus_cat_emb = []
                s_minus_cat_emb = []
                # Since we need to determine the new categorical embedding, we combine the
                # scores here. This is ok, as no cancellations should occur due to tokenization.
                x = ref_as + prev_s_plus - prev_s_minus
                for i_cat in range(len(self.categorical_ids)):
                    # First, compute the reference embedding to compare against
                    x_ref_one_hot = ref_as[
                        :, self.feature_cols_ids == self.categorical_ids[i_cat]
                    ]
                    # TODO: The below code only handles one-hot baselines — it may be interesting
                    #       to also support alternatives like mean embeddings.
                    if torch.allclose(
                        x_ref_one_hot, torch.tensor(0.0).to(x_ref_one_hot)
                    ):
                        # Baseline is zero, so we use an all-zero embedding as fallback
                        x_ref_cat = torch.zeros_like(x_ref_one_hot.argmax(dim=1))
                        ref_emb = torch.zeros_like(
                            self.category_embeddings(
                                x_ref_cat + self.category_idx_offsets[i_cat]
                            )
                        )
                    else:
                        if torch.allclose(x_ref_one_hot.sum(), x_ref_one_hot.max()):
                            raise ValueError(
                                f"Currently, only zero or one-hot baselines are supported, but encountered encoding {x_ref_one_hot}"
                            )
                        x_ref_cat = x_ref_one_hot.argmax(dim=1)
                        ref_emb = self.category_embeddings(
                            x_ref_cat + self.category_idx_offsets[i_cat]
                        )
                    ref_as_cat_emb.append(ref_emb)

                    # Next, compute the positive/negative scores for the given feature.
                    # This is very similar to the standard input layer rule.
                    x_cat_one_hot = x[
                        :, self.feature_cols_ids == self.categorical_ids[i_cat]
                    ]
                    x_cat = x_cat_one_hot.argmax(dim=1)
                    x_cat_emb = self.category_embeddings(
                        x_cat + self.category_idx_offsets[i_cat]
                    )
                    # INFO: Only used in the OOD experiment
                    # if torch.isclose(
                    #     x_cat_one_hot.sum(dim=-1), torch.tensor(0.0)
                    # ).any():
                    #     warnings.warn(
                    #         "FtTokenizer attribution: Multiplying embedding with one-hot sum to model missing features."
                    #     )
                    #     x_cat_emb *= x_cat_one_hot.sum(dim=-1).unsqueeze(-1)
                    # We need to multiply by the one-hot encoding to facilitate the gradient
                    # propagation
                    prev_s_plus_one_hot = prev_s_plus[
                        :, self.feature_cols_ids == self.categorical_ids[i_cat]
                    ]
                    prev_s_minus_one_hot = prev_s_minus[
                        :, self.feature_cols_ids == self.categorical_ids[i_cat]
                    ]
                    gathered_s_plus = torch.gather(
                        prev_s_plus_one_hot, 1, x_cat.unsqueeze(1)
                    )
                    gathered_s_minus = torch.gather(
                        prev_s_minus_one_hot,
                        1,
                        x_ref_cat.unsqueeze(1),
                    )
                    s_plus_cat_emb.append(
                        gathered_s_plus * F.relu(x_cat_emb - ref_emb)
                        + gathered_s_minus * F.relu(ref_emb - x_cat_emb)
                    )
                    s_minus_cat_emb.append(
                        gathered_s_plus * F.relu(ref_emb - x_cat_emb)
                        + gathered_s_minus * F.relu(x_cat_emb - ref_emb)
                    )

                ref_as_cat_emb = torch.stack(ref_as_cat_emb, dim=1)
                s_plus_cat_emb = torch.stack(s_plus_cat_emb, dim=1)
                s_minus_cat_emb = torch.stack(s_minus_cat_emb, dim=1)

                ref_as_emb = torch.cat([ref_as_emb, ref_as_cat_emb], dim=1)
                s_plus_emb = torch.cat([s_plus_emb, s_plus_cat_emb], dim=1)
                s_minus_emb = torch.cat([s_minus_emb, s_minus_cat_emb], dim=1)

            return ref_as_emb, s_plus_emb, s_minus_emb

        self.add_attr_fun(module, attribute_ft_tokenizer)

    def convert_ft_attention(self, module):
        def attribute_ft_attention(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a multi-head attention module.
            """
            if deep:
                warnings.warn(
                    "FtMultiheadAttention: Deep attributions have not yet been implemented for transformers"
                )
            if self.training:
                raise ValueError(
                    "FtMultiheadAttention: Computing attributions while training is not supported"
                )

            expanded_b_V = self.b_V.repeat(1, len(prev_s_plus), 1).reshape(
                len(prev_s_plus), 1, *self.b_V.shape
            )
            expanded_b_V.retain_grad()
            expanded_b_O = self.b_O.repeat(1, len(prev_s_plus)).reshape(
                len(prev_s_plus), 1, *self.b_O.shape
            )
            expanded_b_O.retain_grad()
            self.expanded_bias = [expanded_b_V, expanded_b_O]

            s_plus_v = (
                einsum(
                    prev_s_plus,
                    F.relu(self.W_V),
                    "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
                )
                + einsum(
                    prev_s_minus,
                    F.relu(-self.W_V),
                    "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
                )
                + F.relu(expanded_b_V)
            )
            s_minus_v = (
                einsum(
                    prev_s_plus,
                    F.relu(-self.W_V),
                    "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
                )
                + einsum(
                    prev_s_minus,
                    F.relu(self.W_V),
                    "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
                )
                + F.relu(-expanded_b_V)
            )
            # Need to separately compute new reference activations to avoid adding the bias
            ref_as_v = einsum(
                ref_as,
                self.W_V,
                "b n_feat d_model, n_heads d_model d_head -> b n_feat n_heads d_head",
            )

            # We treat the attention pattern as constant here
            # attn_pattern is (batch, n_heads, n_feat_q, n_feat_k)
            if not torch.allclose(ref_as, torch.zeros_like(ref_as)):
                # Would need to consider the difference in the attention patterns between the reference
                # and the current input — not an issue in practice, as zero reference is the typical
                # use-case.
                raise NotImplementedError(
                    "Attention attribution not yet implemented for non-zero references!"
                )
            attn_pattern = self.compute_attention(
                ref_as + prev_s_plus - prev_s_minus
            ).detach()

            # Note that attention pattern is non-negative, so we don't need to consider flows between signs
            s_plus_v_out = einsum(
                s_plus_v,
                attn_pattern,
                "batch n_feat_k n_heads d_head, batch n_heads n_feat_q n_feat_k -> batch n_feat_q n_heads d_head",
            )
            s_minus_v_out = einsum(
                s_minus_v,
                attn_pattern,
                "batch n_feat_k n_heads d_head, batch n_heads n_feat_q n_feat_k -> batch n_feat_q n_heads d_head",
            )
            ref_as_v_out = einsum(
                ref_as_v,
                attn_pattern,
                "batch n_feat_k n_heads d_head, batch n_heads n_feat_q n_feat_k -> batch n_feat_q n_heads d_head",
            )

            s_plus_out = (
                einsum(
                    s_plus_v_out,
                    F.relu(self.W_O),
                    "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
                )
                + einsum(
                    s_minus_v_out,
                    F.relu(-self.W_O),
                    "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
                )
                + F.relu(expanded_b_O)
            )
            s_minus_out = (
                einsum(
                    s_plus_v_out,
                    F.relu(-self.W_O),
                    "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
                )
                + einsum(
                    s_minus_v_out,
                    F.relu(self.W_O),
                    "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
                )
                + F.relu(-expanded_b_O)
            )
            ref_as_out = einsum(
                ref_as_v_out,
                self.W_O,
                "batch n_feat_q n_heads d_head, n_heads d_head d_model -> batch n_feat_q d_model",
            )

            return ref_as_out, s_plus_out, s_minus_out

        self.add_attr_fun(module, attribute_ft_attention)

    def convert_layernorm(self, module):
        def attribute_layernorm(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a LayerNorm module.
            """
            if deep:
                warnings.warn(
                    "LayerNorm: Deep attributions have not yet been implemented for transformers"
                )

            if self.elementwise_affine:
                weight = self.weight
                bias = self.bias
            else:
                weight = torch.tensor([1.0])
                bias = torch.tensor([0.0])

            if not torch.allclose(ref_as, torch.zeros_like(ref_as)):
                # When references are non-zero, we would need to consider the difference in the
                # LayerNorm weights and biases between the reference and the current input.
                # In practice, using zeroes is the most common use-case.
                raise NotImplementedError(
                    "LayerNorm attribution not yet implemented for non-zero references!"
                )
            var_norm = torch.sqrt(
                torch.var(ref_as + prev_s_plus - prev_s_minus, dim=-1, correction=0)
                + self.eps
            ).unsqueeze(-1)
            aggregate_weight = weight.reshape(1, 1, weight.size(0)) / var_norm
            mean = torch.mean(ref_as + prev_s_plus - prev_s_minus, dim=-1).unsqueeze(-1)
            aggregate_bias = bias.reshape(1, 1, bias.size(0)) - mean * aggregate_weight
            aggregate_weight = aggregate_weight.detach()
            aggregate_bias = aggregate_bias.detach()

            self.expanded_bias = aggregate_bias
            self.expanded_bias.requires_grad = True
            self.expanded_bias.retain_grad()

            s_plus = (
                prev_s_plus * F.relu(aggregate_weight)
                + prev_s_minus * F.relu(-aggregate_weight)
                + F.relu(self.expanded_bias)
            )
            s_minus = (
                prev_s_plus * F.relu(-aggregate_weight)
                + prev_s_minus * F.relu(aggregate_weight)
                + F.relu(-self.expanded_bias)
            )

            new_ref_as = self(ref_as) - bias.reshape(1, 1, bias.size(0))

            return new_ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_layernorm)

    def convert_batchnorm2d(self, module):
        def attribute_batchnorm2d(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a 2D batch norm module.
            """
            if deep:
                warnings.warn(
                    "BatchNorm2d: Deep attributions have not yet been implemented for CNNs"
                )

            if self.training:
                raise ValueError(
                    "BatchNorm2d: Computing attributions while training is not supported"
                )
            if not self.track_running_stats:
                raise NotImplementedError(
                    "BatchNorm2d: Computing attributions for batch norm without running stats not implemented"
                )
            if self.affine:
                weight = self.weight
                bias = self.bias
            else:
                weight = 1
                bias = 0

            var_norm = torch.sqrt(self.running_var + self.eps)
            aggregate_weight = (
                (weight / var_norm).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            )
            aggregate_bias = (
                (bias - self.running_mean * weight / var_norm)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            n_batch, n_channel, h, w = prev_s_plus.shape
            self.expanded_bias = aggregate_bias.repeat(n_batch, 1, 1, 1)
            self.expanded_bias.retain_grad()

            s_plus = (
                prev_s_plus * F.relu(aggregate_weight)
                + prev_s_minus * F.relu(-aggregate_weight)
                + F.relu(self.expanded_bias)
            )
            s_minus = (
                prev_s_plus * F.relu(-aggregate_weight)
                + prev_s_minus * F.relu(aggregate_weight)
                + F.relu(-self.expanded_bias)
            )

            new_ref_as = ref_as * aggregate_weight + aggregate_bias
            if not torch.allclose(new_ref_as, self(ref_as), atol=1e-5, rtol=1e-4):
                raise RuntimeError(
                    "BatchNorm2d: Wrapped BatchNorm2d behaviour differed from the original module"
                )

            return new_ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_batchnorm2d)

    def convert_error(self, module):
        def attribute_error(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Raises error when trying to propagate through an unsupported module.
            """
            raise ValueError(f"Should not be trying to propagate scores through {self}")

        self.add_attr_fun(module, attribute_error)

    def convert_residual(self, module):
        def attribute_residual(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            module_ref_as, module_s_plus, module_s_minus = self.block.attribute_forward(
                ref_as, prev_s_plus, prev_s_minus, deep=False
            )
            identity_ref_as, identity_s_plus, identity_s_minus = (
                self.identity.attribute_forward(
                    ref_as, prev_s_plus, prev_s_minus, deep=False
                )
            )
            ref_as = module_ref_as + identity_ref_as
            s_plus = module_s_plus + identity_s_plus
            s_minus = module_s_minus + identity_s_minus
            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_residual)

    def convert_slice(self, module):
        def attribute_slice(self, ref_as, prev_s_plus, prev_s_minus, deep=False):
            """
            Propagates attribution scores through a slice module.
            """
            s_plus = self(prev_s_plus)
            s_minus = self(prev_s_minus)
            ref_as = self(ref_as)

            return ref_as, s_plus, s_minus

        self.add_attr_fun(module, attribute_slice)

    # TODO: Should instead define bias collectors per module to make this more extensible.
    def _collect_bias_attrs(self, module, x, retain_bias=False):
        bias_attrs = []
        for layer in module:
            if isinstance(layer, ResidualBlock):
                residual_module_bias = self._collect_bias_attrs(
                    layer.block, x, retain_bias=retain_bias
                )
                if not isinstance(layer.identity, nn.Identity):
                    raise ValueError(
                        "Encountered non-identity identity block in residual."
                    )

                bias_attrs.append(residual_module_bias)
            elif hasattr(layer, "expanded_bias") and layer.expanded_bias is not None:
                if isinstance(layer.expanded_bias, list):
                    expanded_bias = layer.expanded_bias
                else:
                    expanded_bias = [layer.expanded_bias]

                for bias_elem in expanded_bias:
                    bias_attrs.append(
                        (bias_elem.grad * bias_elem).flatten(start_dim=1).sum(dim=-1)
                    )
                    del bias_elem.grad
                if not retain_bias:
                    del layer.expanded_bias
        if not bias_attrs:
            return torch.zeros(len(x)).to(x.device)
        return sum(bias_attrs)

    def _collect_deep_scores(self, retain_scores=False):
        deep_attrs = []
        for layer in self.model:
            if hasattr(layer, "prev_s_plus") and hasattr(layer, "prev_s_minus"):
                deep_attr = (
                    layer,
                    layer.prev_s_plus.grad * layer.prev_s_plus
                    + layer.prev_s_minus.grad * layer.prev_s_minus,
                )
                if hasattr(layer, "expanded_bias") and layer.expanded_bias is not None:
                    deep_attr += (layer.expanded_bias.grad * layer.expanded_bias,)
                else:
                    deep_attr += (None,)
                deep_attrs.append(deep_attr)
                del layer.prev_s_plus.grad
                del layer.prev_s_minus.grad
                if not retain_scores:
                    del layer.prev_s_plus
                    del layer.prev_s_minus
        return deep_attrs

    def attribute(self, x, target, ref=None, deep=False):
        """
        Computes attribution scores for the wrapped model.

        Directly implements the input layer rule, delegates the computation of the other rules to submodules of the model.
        """
        flat_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            flat_input = True
        # Input is now (*, num_features)

        if ref is None:
            # TODO: It might be better to use shape (1, ...) and broadcast,
            #       but this doesn't work with FT-Transformer tokenizer
            ref = torch.zeros_like(x)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        # Reference is now (1, num_features)

        x = x.detach() - ref.detach()
        x.requires_grad = True
        x.retain_grad()

        # Shape of (*, num_neurons == num_features)
        s_plus = F.relu(x)
        s_minus = F.relu(-x)
        ref_as = ref

        # Forward pass
        _, s_plus, s_minus = self.model.attribute_forward(
            ref_as, s_plus, s_minus, deep=deep
        )

        # Backward pass
        # TODO: The code for handling target dimensionality seems potentially brittle — consider refactoring
        # First, ensure that the target index is a non-0-dimensional tensor
        if not isinstance(target, torch.Tensor):
            target_index = torch.tensor(target)
        else:
            target_index = target
        if target_index.dim() == 0:
            target_index = target_index.unsqueeze(0)

        # Next, expand target index to match the number of samples (as gather does not broadcast)
        if len(target_index) == 1 and len(x) > 1:
            target_index = target_index.repeat(len(x), 1)
        elif len(target_index) == len(x):
            target_index = target_index
        else:
            raise ValueError(
                "Target index must be scalar or match the number of samples"
            )
        while target_index.dim() < s_plus.dim():
            target_index = target_index.unsqueeze(-1)
        target_index = target_index.to(x.device)
        torch.autograd.backward(
            s_plus.gather(-1, target_index).split(1, dim=0), retain_graph=True
        )
        feat_attrs_plus = x.grad * x
        if deep:
            deep_scores_plus = self._collect_deep_scores(retain_scores=True)
        bias_attrs_plus = self._collect_bias_attrs(self.model, x, retain_bias=True)
        del x.grad
        torch.autograd.backward(
            s_minus.gather(-1, target_index).split(1, dim=0), retain_graph=False
        )
        feat_attrs_minus = x.grad * x
        if deep:
            deep_scores_minus = self._collect_deep_scores(retain_scores=False)
        bias_attrs_minus = self._collect_bias_attrs(self.model, x, retain_bias=False)
        del x.grad

        assert (feat_attrs_plus >= -1e-9).all(), "feat_attrs_plus has negative values"
        assert (feat_attrs_minus >= -1e-9).all(), "feat_attrs_minus has negative values"
        assert (bias_attrs_plus >= -1e-9).all(), "bias_attrs_plus has negative values"
        assert (bias_attrs_minus >= -1e-9).all(), "bias_attrs_minus has negative values"

        feat_attrs = (feat_attrs_plus.detach(), feat_attrs_minus.detach())
        bias_attrs = (bias_attrs_plus.detach(), bias_attrs_minus.detach())

        if flat_input:
            feat_attrs = tuple(fa.squeeze(0) for fa in feat_attrs)
            bias_attrs = tuple(ba.squeeze(0) for ba in bias_attrs)

        if not deep:
            return feat_attrs, bias_attrs
        else:
            return feat_attrs, bias_attrs, (deep_scores_plus, deep_scores_minus)
